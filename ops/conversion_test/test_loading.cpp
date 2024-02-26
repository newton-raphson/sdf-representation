#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);

        // Enable gradient computation for the module
        module.eval();

        // Read input from input.csv
        std::ifstream input_file("input.csv");
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
            std::cout << "row: " << row << '\n';
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
        torch::autograd::Variable input_variable(input_tensor, true);

        // Execute the model and get the output variable
        torch::autograd::Variable output_variable = module.forward({input_variable}).toTensor();

        // Read target output from output.csv
        std::ifstream output_file("output.csv");
        if (!output_file.is_open()) {
            std::cerr << "Error opening output file\n";
            return -1;
        }

        // Read target output data from the CSV file
        std::vector<float> target_output_data;
        while (std::getline(output_file, line)) {
            target_output_data.push_back(std::stof(line));
        }

        // Close the output file
        output_file.close();

        // Convert target_output_data to a torch::Tensor
        torch::Tensor target_output_tensor = torch::empty({target_output_data.size()}, torch::kFloat32);

        for (size_t i = 0; i < target_output_data.size(); ++i) {
            target_output_tensor[i] = target_output_data[i];
        }

        // Wrap the target output tensor in a torch::autograd::Variable for gradient computation
        torch::autograd::Variable target_output_variable(target_output_tensor, true);

        // Compute the mean squared error loss
        torch::Tensor loss_tensor = torch::mse_loss(output_variable, target_output_variable);

        // Backward pass to compute gradients
        loss_tensor.backward();

        // Get the gradients from the input_variable
        torch::Tensor gradient = input_variable.grad();

        // Convert the gradient to a std::vector for printing
        std::vector<float> gradient_vector(gradient.data_ptr<float>(), gradient.data_ptr<float>() + gradient.numel());

        // Print the gradient vector
        std::cout << "Gradient: " << gradient_vector << '\n';
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << '\n';
        return -1;
    }

    std::cout << "ok\n";

    return 0;
}
