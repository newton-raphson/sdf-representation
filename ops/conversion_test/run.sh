
cd build/
rm -r *
cmake ../.
make
cd ..
./build/test_loading implicit_model.pt