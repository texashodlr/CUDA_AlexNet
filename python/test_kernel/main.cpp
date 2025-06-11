#include <torch/extension.h>

std::string hello_world(){
    return "As you can see, your mum gay";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("hello_world", torch::wrap_pybind_function(hello_world), "hello_world");
}