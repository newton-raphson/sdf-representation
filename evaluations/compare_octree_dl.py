import vtk
import torch
import numpy as np




# Read PVTU file
reader = vtk.vtkXMLPUnstructuredGridReader()
reader.SetFileName('your_mesh.pvtu')
reader.Update()

# Get the output of the reader
output = reader.GetOutput()

# Get the number of points (nodes) in the mesh
num_points = output.GetNumberOfPoints()

# Loop through each node in the mesh
for i in range(num_points):
    # Get coordinates of the current node
    coordinates = np.array(output.GetPoint(i))

    # Preprocess the coordinates (adjust as needed)
    preprocessed_coordinates = transform(coordinates)

    # Convert to PyTorch tensor
    input_tensor = torch.Tensor(preprocessed_coordinates).unsqueeze(0)

    # Pass the coordinates through the pre-trained neural network
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Process the output as needed
    # Example: Print the output tensor
    print(f"Node {i + 1}, Output Tensor: {output_tensor}")
