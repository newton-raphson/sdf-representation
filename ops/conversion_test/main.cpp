#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <torch/csrc/autograd/autograd.h>
#include <cmath>
class ImplicitNetCompatible : public torch::nn::Module {
public:
    ImplicitNetCompatible(
        int d_in,
        std::vector<int64_t> dims,
        std::vector<int64_t> skip_in,
        bool geometric_init=true,
        double radius_init=1,
        double beta=99
    );

    torch::Tensor forward(torch::Tensor input);

private:
    int num_layers;
    std::vector<int64_t> skip_in;
    torch::nn::ModuleList layers;
    torch::nn::Softplus activation;
};

ImplicitNetCompatible::ImplicitNetCompatible(
    int d_in,
    std::vector<int64_t> dims,
    std::vector<int64_t> skip_in,
    bool geometric_init,
    double radius_init,
    double beta
) : num_layers(dims.size()), skip_in(skip_in), activation(torch::nn::SoftplusOptions().beta(beta)) {

    dims.insert(dims.begin(), d_in);
    dims.push_back(1);

    for (int layer = 0; layer < num_layers - 1; ++layer) {
        int64_t out_dim = (std::find(skip_in.begin(), skip_in.end(), layer + 1) != skip_in.end()) ?
                          dims[layer + 1] - d_in : dims[layer + 1];

        auto lin = register_module("layers" +std::to_string(layer), torch::nn::Linear(dims[layer], out_dim));

        if (geometric_init) {
            if (layer == num_layers - 2) {
                torch::nn::init::normal_(lin->weight, std::sqrt(3.1415926)/ std::sqrt(static_cast<double>(dims[layer])));
                torch::nn::init::constant_(lin->bias, -radius_init);
            } else {
                torch::nn::init::constant_(lin->bias, 0.0);
                torch::nn::init::normal_(lin->weight, 0.0, std::sqrt(2) / std::sqrt(static_cast<double>(out_dim)));
            }
        }
        layers->push_back(lin);
    }
}

torch::Tensor ImplicitNetCompatible::forward(torch::Tensor input) {
    torch::Tensor x = input;

    for (int layer = 0; layer < num_layers - 1; ++layer) {
        if (std::find(skip_in.begin(), skip_in.end(), layer) != skip_in.end()) {
            x = torch::cat({x, input}, -1) / std::sqrt(2);
        }

        x = layers[layer]->as<torch::nn::Linear>()->forward(x); ;

        if (layer < num_layers - 2) {
            x = activation(x);
        }
    }

    return x;
}

int main(int argc, const char* argv[] ) {
    
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }
    // Load the model from the saved .pt file
    torch::jit::script::Module scripted_model;
    
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        scripted_model = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    // // Example input tensor
    // torch::Tensor input = torch::randn({1, 10}).requires_grad(true);

    // Instantiate the C++ model
    ImplicitNetCompatible cpp_model(3,{512,512,512,512,512,512,512,512},{4},true,1,99);


    // std::cout<<cpp_model.named_parameters()<<std::endl;
    // std::cout<<scripted_model.named_parameters()<<std::endl;
    auto named_params = cpp_model.named_parameters(/* recurse */ true);
    std::cout << "Named parameters in the C++ model\n";
    // print the named parameters
    for (auto pair : named_params) {
        
        std::cout << pair.key() << std::endl;
    }
    
// Load parameters from the scripted model
for (auto pair : scripted_model.named_parameters()) {
    // torch::Tensor tensor = pair->value().toTensor().detach().clone();
    
    // pair.value
    torch::Tensor tensor = pair.value.clone();
   
    std::string name = pair.name;
    // name.erase(0, 7);  // Remove "module." prefix added during scripting
        // Check if the parameter exists
    // layers
    size_t dotPos = name.find('.');
    std::string parameter_name = name.substr(dotPos + 1);
    // std::cout << "parameter_name: " << parameter_name << std::endl;
    
    std::replace(name.begin(), name.begin()+7, '.', '\0');
    std::cout << "name: " << name << std::endl;
    torch::Tensor* tensor_new = cpp_model.named_parameters().find(name);
    tensor_new->copy_(tensor);
    // named_params.find(name)->value().data().copy_(tensor);
    // if (named_params.contains(name)) {
    //     // Get the tensor from the named parameters
    //     torch::Tensor& target_tensor = named_params[name];

    //     // Modify the values of the tensor
    //     // For example, fill it with zeros
    //     target_tensor = tensor.clone();
    //     printf("Parameter found: %s\n", name.c_str());
    // }
    // else
    // {
    //     std::cout << "Parameter not found: " << name << std::endl;
    // }
     
    // cpp_model.named_parameters().find(name)->data().copy_(tensor);
    // cpp_model.named_parameters()
    // std::cout << "Model loaded successfully\n";
    // cpp_model.get_parameter(name).data().copy_(tensor);
}
    cpp_model.eval();
    
    // std::cout << "Model loaded successfully\n";

    // get the input tensor from csv
    // Read input data from the CSV file
    std::ifstream input_file("input.csv");
    if (!input_file.is_open()) {
        std::cerr << "Error opening input file\n";
        return -1;
    }

    std::string line;
    std::vector<std::vector<float>> input_data;

    while (std::getline(input_file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stof(cell));
        }
        // std::cout << "row: " << row << '\n';
        input_data.push_back(row);
    }
    // Close the input file
    input_file.close();

        // Convert input data to torch tensor
    torch::Tensor input = torch::from_blob(input_data.data(), {input_data.size(), input_data[0].size()}).clone();
    input.set_requires_grad(true);

    // Forward pass
    torch::Tensor output = cpp_model.forward(input);

    // Compute gradients using torch::autograd::grad
    torch::Tensor grad_output = torch::ones_like(input);
    std::vector<torch::Tensor> gradients = torch::autograd::grad({output}, {input}, {grad_output});

    // Print gradients
    std::cout << "Gradients with respect to input: " << gradients[0] << std::endl;

    return 0;
}
