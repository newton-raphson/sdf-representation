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
        // Wrap the input tensor in a vector of IValue.
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        // std::cout << "input_tensor: " << input_tensor << '\n';

        // Execute the model and turn its output into a tensor.
        at::Tensor output = module.forward(inputs).toTensor();

        // std::cout << "output: " << output << '\n';
        // Read output from output.csv
        std::ifstream output_file("output.csv");
        if (!output_file.is_open()) {
            std::cerr << "Error opening output file\n";
            return -1;
        }

        // Read output data from the CSV file
        std::vector<float> output_data;
        while (std::getline(output_file, line)) {
            output_data.push_back(std::stof(line));
        }

        // Close the output file
        output_file.close();

        // Convert output_data to a torch::Tensor
         
        torch::Tensor target_output_tensor = torch::empty({output_data.size()}, torch::kFloat32);

        for (size_t i = 0; i < output_data.size(); ++i) {
            target_output_tensor[i] = output_data[i];
        }
         std::cout << "target_output_tensor: " << target_output_tensor << '\n';
        // Compute the difference between the model output and the target output
        torch::Tensor difference = output - target_output_tensor;

        // // Print the difference
        std::cout << "Difference between model output and target output:\n" << difference << '\n';

        // Save the difference to a difference.csv file
        std::ofstream difference_file("difference.csv");
        if (!difference_file.is_open()) {
            std::cerr << "Error opening difference file for writing\n";
            return -1;
        }

        // Convert the difference tensor to a vector for writing to the file
        std::vector<float> difference_vector(difference.data_ptr<float>(), difference.data_ptr<float>() + difference.numel());

        // Write the difference vector to the file
        for (float value : difference_vector) {
            difference_file << value << '\n';
        }

        // Close the difference file
        difference_file.close();
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << '\n';
        return -1;
    }

    std::cout << "ok\n";

    return 0;
}
