import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
std::string hello_world(){
    return "As you can see, your mum gay";
}
"""

test_kernel = load_inline(
        name = 'test_kernel',
        cpp_sources=[cpp_source],
        functions=['hello_world'],
        verbose=True,
        build_directory='./test_kernel'
)

print(test_kernel.hello_world())

