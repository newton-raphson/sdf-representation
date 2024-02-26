//
// Created by samundrakarki on 02/22/24.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <onnxruntime_cxx_api.h>
#include <vector>
// #include <torch/script.h>


/// TEST CODE TO TAKE INPUT FROM CSV FILE AND PASS TO ONNX MODEL
/// AND GET THE OUTPUT FROM THE MODEL WITH THE GRADIENTS OF THE INPUT

int main(int argc, char *argv[]) {
    // Initialize the ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    // Set up session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    // Create a session with the loaded model
    Ort::Session session(env, argv[1], session_options);

    // Define names of the input and output tensors
    const char* input_name = "input"; // Replace with the actual input name obtained from the model
    const char* output_name = "output"; // Replace with the actual output name obtained from the model

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::RunOptions run_options{nullptr};

    // Read input data from CSV file
    std::ifstream input_file("input.csv");
    std::vector<std::vector<float>> input_data;
    std::string line;
    while (std::getline(input_file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        float value;
        while (ss >> value) {
            row.emplace_back(value);
        }
        input_data.emplace_back(row);
    }
    input_file.close();

    // Initialize gradients
    std::vector<std::vector<float>> output_gradients(input_data.size(), std::vector<float>(input_data[0].size(), 0.0));
    std::vector<std::vector<float>> outputs(input_data.size(), std::vector<float>());

    // Iterate over the input data
    for (size_t i = 0; i < input_data.size(); i++) {
        // Prepare input tensor
        std::vector<float> input_tensor_values = input_data[i];
        std::vector<int64_t> input_tensor_shape = {1, static_cast<int64_t>(input_tensor_values.size())};

        // Reuse memory for input and output tensors
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());

        // Perform the inference
        Ort::Value output_tensor = std::move(session.Run(run_options, &input_name, &input_tensor, 1, &output_name, 1).front());
        float* floatarr = output_tensor.GetTensorMutableData<float>();
        outputs[i] = std::vector<float>(floatarr, floatarr + output_tensor.GetTensorTypeAndShapeInfo().GetElementCount());

        // Compute the gradients
        // output_tensor.SetIsTensorAndOwnsStorage(false);
        std::vector<Ort::Value> output_gradient_tensors;
        session.Run(run_options, &output_name, &output_tensor, 1, nullptr, 0, &output_gradient_tensors);

        // Accumulate the gradients
        for (size_t j = 0; j < input_data[i].size(); j++) {
            output_gradients[i][j] += output_gradient_tensors[0].GetTensorMutableData<float>()[j];
        }
    }

    // Save the gradients and outputs to CSV files
    std::ofstream output_file("output.csv");
    std::ofstream gradient_file("output_gradients.csv");
    for (size_t i = 0; i < outputs.size(); i++) {
        for (size_t j = 0; j < outputs[i].size(); j++) {
            output_file << outputs[i][j] << ",";
        }
        output_file << "\n";
    }
    for (size_t i = 0; i < output_gradients.size(); i++) {
        for (size_t j = 0; j < output_gradients[i].size(); j++) {
            gradient_file << output_gradients[i][j] << ",";
        }
        gradient_file << "\n";
    }
    output_file.close();
    gradient_file.close();

    return 0;
}