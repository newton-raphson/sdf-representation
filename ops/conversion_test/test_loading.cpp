#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <torch/csrc/autograd/autograd.h>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }
    // std::cout << "LibTorch version: " << torch::version() << std::endl;
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);

        // Enable gradient computation for the module
        module.eval();

        // Read input from input.csv
        std::ifstream input_file("nodes_coordinates.csv");
        if (!input_file.is_open()) {
            std::cerr << "Error opening input file\n";
            return -1;
        }

        // Read input data from the CSV file
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

        // Convert input_data to a torch::Tensor
        torch::Tensor input_tensor = torch::empty({input_data.size(), input_data[0].size()}, torch::kFloat32);

        for (size_t i = 0; i < input_data.size(); ++i) {
            for (size_t j = 0; j < input_data[0].size(); ++j) {
                input_tensor[i][j] = input_data[i][j];
            }
        }

        // Wrap the input tensor in a torch::autograd::Variable for gradient computation
        torch::autograd::Variable input_variable(input_tensor);
        input_variable.set_requires_grad(true);
        std::cout << "input_variable: " << input_variable << '\n';
        // Execute the model and get the output variable
        torch::autograd::Variable output_variable = module.forward({input_variable}).toTensor();
        std::cout << "output_variable: " << output_variable << '\n';

        // save the output in a csv file
        std::ofstream output_file("output.csv");

        if (!output_file.is_open()) {
            std::cerr << "Error opening output file\n";
            return -1;
        }
        else {
            // Iterate over the elements of the tensor and save to the file
            for (size_t i = 0; i < output_variable.numel(); ++i) {
                output_file << output_variable.index({static_cast<int64_t>(i)}).item<float>() << ',';
                output_file << '\n';
            }
            
            // Close the file after writing
            output_file.close();
        }

        output_variable.set_requires_grad(true);
        if (!output_variable.requires_grad()) {
            std::cerr << "Error: Gradient computation is not enabled for target_output_variable\n";
            return -1;
        }
                // torch::Tensor gradient_output = target_output_variable.grad();
        torch::Tensor gradient = torch::ones_like(input_variable);
        // std::cout << "gradient: " << gradient << '\n';

        // // create a loss function and first perform a forward pass
        // torch::Tensor loss = torch::nn::functional::mse_loss(output_variable, output_variable);

        // // perform a backward pass
        // loss.backward();
        // this creates the backward graph and computes the gradients
        // calculate the gradient of output with respect to the input
        std::vector<torch::Tensor> gradients = torch::autograd::grad({output_variable}, {input_variable}, {gradient}, /*retain_graph=*/true);

    
    //     std::cout << "gradient: " << gradient << '\n';
        // normalize the gradient output
        // Normalize the gradient tensor
        std::ofstream gradient_file("gradient.csv");
        for (size_t i = 0; i < gradients.size(); ++i) {
            // torch::Tensor row = gradients[i];
            // float norm_factor = row.norm().item<float>();
            
            // // Check if the norm factor is not zero to avoid division by zero
            // if (norm_factor != 0.0) {
            //     // Normalize the row
            //     gradients[i] = row / norm_factor;
            // }
            // Write the normalized row to the file
                // Get a pointer to the tensor's data
            const float* data_ptr = gradients[i].data_ptr<float>();

            // Get the total number of elements in the tensor
            int num_elements = gradients[i].numel();

            // Access individual values
            for (int i = 0; i < 3; ++i) {
                float value = data_ptr[i];
                gradient_file << value << ',';
            }
            gradient_file << '\n';
            
        }
        gradient_file.close();
        std::cout <<"Gradients Calculated\n";
        std::cout << "gradient: " << gradients << '\n';

        



        // save the gradient in a csv file
        // std::ofstream gradient_file("gradient.csv");
        // if (!gradient_file.is_open()) {
        //     std::cerr << "Error opening gradient file\n";
        //     return -1;
        // }
        // else {
        //     // Iterate over the elements of the tensor and save to the file
        //     for (size_t i = 0; i < gradients.size(); ++i) {
        //         for (size_t j = 0; j < gradients[i].numel(); ++j) {
        //             gradient_file << gradients[i].index({static_cast<int64_t>(j)}).item<float>() << ',';
        //         }
        //         gradient_file << '\n';
        //     }

        //     // Close the file after writing
        //     gradient_file.close();
        // }

        // // Read target output from output.csv
        // std::ifstream output_file("output.csv");
        // if (!output_file.is_open()) {
        //     std::cerr << "Error opening output file\n";
        //     return -1;
        // }

        // // Read target output data from the CSV file
        // std::vector<float> target_output_data;
        // while (std::getline(output_file, line)) {
        //     target_output_data.push_back(std::stof(line));
        // }

        // // Close the output file
        // output_file.close();

        // // Convert target_output_data to a torch::Tensor
        // torch::Tensor target_output_tensor = torch::empty({target_output_data.size()}, torch::kFloat32);

        // for (size_t i = 0; i < target_output_data.size(); ++i) {
        //     target_output_tensor[i] = target_output_data[i];
        // }


        std::cout << "##################################################################\n";
        std::cout << "##############################OKAY################################\n";
        std::cout << "##################################################################\n";

    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << '\n';
        return -1;
    }

    std::cout << "ok\n";

    return 0;
}
