"""
This script compares the public interfaces of the `torch.cuda` and `torch.gcu` modules and identifies unsupported interfaces in `torch.gcu`. It provides functionalities to generate documentation in various formats (reStructuredText, Excel, CSV) and print a formatted table of the comparison results.

Functions:
    get_public_members(module):
        Retrieves the public members of a given module based on the module's `__all__` attribute.

    is_module(obj):
        Checks if the given object is a module.

    cuda_name_to_gcu_name(name):

    compare_modules(cuda_module, gcu_module, current_cuda_path=None, current_gcu_path=None, max_depth=5, current_depth=0, visited=None):
        Compares the public interface of two modules (torch.cuda and torch.gcu) and identifies unsupported interfaces.

    compare_cuda_gcu_interfaces():
        Compares the interfaces of torch.cuda and torch.gcu modules.

    generate_rst_from_mapping(interface_mapping, unsupported_interfaces):
        Generates reStructuredText (reST) content from the given interface mappings.

    generate_excel_from_mapping(interface_mapping, unsupported_interfaces, excel_path):

    generate_csv_from_mapping(interface_mapping, unsupported_interfaces, csv_path):

    print_pretty_table(interface_mapping, unsupported_interfaces):

    main():
        The main function that parses command-line arguments and performs the comparison, generating the requested output.

"""

import argparse
import csv
import pandas as pd
from tabulate import tabulate
import torch
import torch_gcu


def get_public_members(module):
    """
    Retrieves the public members of a given module.

    This function filters the members of the provided module based on whether
    they are listed in the module's `__all__` attribute, which typically 
    defines the public API of the module.

    Args:
        module (module): The module from which to retrieve public members.

    Returns:
        dict: A dictionary where the keys are the names of the public members
              and the values are the corresponding attributes from the module.
    """

    def filter(name):
        return hasattr(module, '__all__') and name in module.__all__

    return {
        name: getattr(module, name)
        for name in dir(module) if filter(name)
    }


def is_module(obj):
    """
    Check if the given object is a module.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is a module, False otherwise.
    """
    return isinstance(obj, type(torch.cuda))


def cuda_name_to_gcu_name(name):
    """
    Converts CUDA-related names to GCU-related names.

    This function replaces occurrences of 'cuda' with 'gcu', 'CUDA' with 'GCU',
    and 'nv' with 'tops' in the given string.

    Args:
        name (str): The original name containing CUDA-related terms.

    Returns:
        str: The modified name with GCU-related terms.
    """
    # TODO: add nccl/NCCL to eccl/ECCL
    return name.replace('cuda', 'gcu').replace('CUDA',
                                               'GCU').replace('nv', 'tops')


def compare_modules(cuda_module,
                    gcu_module,
                    current_cuda_path=None,
                    current_gcu_path=None,
                    max_depth=5,
                    current_depth=0,
                    visited=None):
    """
    Compares the public interface of two modules (torch.cuda and torch/gcu) and identifies unsupported interfaces.
        Args:
            cuda_module (module): The CUDA module to compare.
            gcu_module (module): The GCU module to compare.
            current_cuda_path (str, optional): The current path in the CUDA module hierarchy. Defaults to None.
            current_gcu_path (str, optional): The current path in the GCU module hierarchy. Defaults to None.
            max_depth (int, optional): The maximum depth to recurse into the module hierarchy. Defaults to 5.
            current_depth (int, optional): The current depth in the module hierarchy. Defaults to 0.
            visited (set, optional): A set of visited modules to prevent infinite recursion. Defaults to None.
        Returns:
            tuple: A tuple containing:
                - dict: A mapping of GCU interface paths to CUDA interface paths.
                - list: A list of unsupported CUDA interface paths.
    """
    if visited is None:
        visited = set()

    # Prevent infinite recursion
    if current_depth > max_depth or cuda_module in visited or gcu_module in visited:
        return {}, []

    visited.add(cuda_module)
    visited.add(gcu_module)

    cuda_members = get_public_members(cuda_module)
    gcu_members = get_public_members(gcu_module)

    interface_mapping = {}
    unsupported_interfaces = []

    if not current_cuda_path:
        current_cuda_path = cuda_module.__name__
    if not current_gcu_path:
        current_gcu_path = gcu_module.__name__

    for member_name, cuda_member in cuda_members.items():
        gcu_member_name = cuda_name_to_gcu_name(member_name)
        gcu_member = gcu_members.get(gcu_member_name)

        new_current_cuda_path = f"{current_cuda_path}.{member_name}"
        new_current_gcu_path = f"{current_gcu_path}.{gcu_member_name}"

        if gcu_member is None:
            unsupported_interfaces.append(new_current_cuda_path)
        elif callable(cuda_member) and callable(gcu_member):
            interface_mapping[new_current_gcu_path] = new_current_cuda_path
        elif is_module(cuda_member) and is_module(gcu_member):
            sub_mapping, sub_unsupported = compare_modules(
                cuda_member,
                gcu_member,
                current_cuda_path=new_current_cuda_path,
                current_gcu_path=new_current_gcu_path,
                max_depth=max_depth,
                current_depth=current_depth + 1,
                visited=visited)
            interface_mapping.update(sub_mapping)
            unsupported_interfaces.extend(sub_unsupported)
        else:
            interface_mapping[new_current_gcu_path] = new_current_cuda_path

    return interface_mapping, unsupported_interfaces


def compare_cuda_gcu_interfaces():
    """
    Compare the interfaces of torch.cuda and torch.gcu modules.

    This function checks if the torch.gcu module is available. If it is not available,
    it returns an empty dictionary and an empty list. If torch.gcu is available, it
    compares the interfaces of the torch.cuda and torch.gcu modules using the 
    compare_modules function.

    Returns:
        tuple: A tuple containing a dictionary of differences and a list of missing 
               interfaces.
    """
    if torch.gcu is None:
        return {}, []
    return compare_modules(torch.cuda, torch.gcu, "torch.cuda", "torch.gcu")


def generate_rst_from_mapping(interface_mapping,
                              unsupported_interfaces,
                              gen_unsupported=False):
    """
    Generate reStructuredText (reST) content from the given interface mappings.

    This function creates a reST document that lists supported and unsupported
    interfaces between `torch.gcu` and `torch.cuda`. The generated document
    includes tables for both supported and unsupported interfaces.

    Args:
        interface_mapping (dict): A dictionary where keys are `torch.gcu` interfaces
                                  and values are corresponding `torch.cuda` interfaces.
        unsupported_interfaces (list): A list of `torch.cuda` interfaces that are not
                                       supported in `torch.gcu`.

    Returns:
        str: A string containing the generated reST content.
    """

    rst_content = []

    rst_content.append(".. _python_op:")
    rst_content.append("")
    rst_content.append("####################")
    rst_content.append("Python接口支持情况")
    rst_content.append("####################")
    rst_content.append("")

    # Title for supported interfaces
    rst_content.append("Supported Interfaces Mapping")
    rst_content.append("===========================")
    rst_content.append("")

    if interface_mapping:
        rst_content.append(".. list-table:: Supported Interfaces")
        rst_content.append("   :header-rows: 1")
        rst_content.append("")
        rst_content.append("   * - torch.gcu Interface")
        rst_content.append("     - torch.cuda Interface")
        rst_content.append("")

        for gcu_interface, cuda_interface in sorted(interface_mapping.items()):
            rst_content.append(f"   * - {gcu_interface}")
            rst_content.append(f"     - {cuda_interface}")
        rst_content.append("")

    else:
        rst_content.append("No supported interfaces found.")
        rst_content.append("")

    if gen_unsupported:
        # Title for unsupported interfaces
        rst_content.append("Unsupported Interfaces in torch.gcu")
        rst_content.append("==================================")
        rst_content.append("")
        if unsupported_interfaces:
            rst_content.append(".. list-table:: Unsupported Interfaces")
            rst_content.append("   :header-rows: 0")
            rst_content.append("")

            for interface in sorted(unsupported_interfaces):
                rst_content.append(f"   * - {interface}")
            rst_content.append("")

        else:
            rst_content.append(
                "All torch.cuda interfaces are supported in torch.gcu.")
            rst_content.append("")
    return '\n'.join(rst_content)


def generate_excel_from_mapping(interface_mapping, unsupported_interfaces,
                                excel_path):
    """
    Generates an Excel file from the given interface mappings and unsupported interfaces.

    Args:
        interface_mapping (dict): A dictionary where keys are 'torch.gcu' interfaces and values are 'torch.cuda' interfaces.
        unsupported_interfaces (list): A list of unsupported interface names.
        excel_path (str): The file path where the Excel file will be saved.

    Returns:
        None
    """
    supported_df = pd.DataFrame(
        list(interface_mapping.items()),
        columns=['torch.gcu Interface', 'torch.cuda Interface'])
    unsupported_df = pd.DataFrame(unsupported_interfaces,
                                  columns=['Unsupported Interface'])

    with pd.ExcelWriter(excel_path) as writer:
        supported_df.to_excel(writer,
                              sheet_name='Supported Interfaces',
                              index=False)
        unsupported_df.to_excel(writer,
                                sheet_name='Unsupported Interfaces',
                                index=False)


def generate_csv_from_mapping(interface_mapping, unsupported_interfaces,
                              csv_path):
    """
    Generates a CSV file from the provided interface mappings and unsupported interfaces.

    Args:
        interface_mapping (dict): A dictionary where keys are 'torch.gcu' interfaces and values are corresponding 'torch.cuda' interfaces.
        unsupported_interfaces (list): A list of unsupported interface names.
        csv_path (str): The file path where the CSV file will be saved.

    Returns:
        None
    """
    supported_data = [['torch.gcu Interface', 'torch.cuda Interface']] + list(
        interface_mapping.items())
    unsupported_data = [['Unsupported Interface']
                        ] + [[item] for item in unsupported_interfaces]

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(supported_data)
        writer.writerow([])
        writer.writerows(unsupported_data)


def print_pretty_table(interface_mapping, unsupported_interfaces):
    """
    Prints a formatted table of supported and unsupported interfaces.

    Args:
        interface_mapping (dict): A dictionary where keys are 'torch.gcu' interfaces 
                                  and values are corresponding 'torch.cuda' interfaces.
        unsupported_interfaces (list): A list of unsupported 'torch.gcu' interfaces.

    Returns:
        None
    """
    supported_data = [['torch.gcu Interface', 'torch.cuda Interface']] + list(
        interface_mapping.items())
    unsupported_data = [['Unsupported Interface']
                        ] + [[item] for item in unsupported_interfaces]

    print("Supported Interfaces:")
    print(tabulate(supported_data, headers="firstrow", tablefmt="grid"))
    print("\nUnsupported Interfaces in torch.gcu:")
    print(tabulate(unsupported_data, headers="firstrow", tablefmt="grid"))


def main():
    parser = argparse.ArgumentParser(
        description="Compare torch.cuda and torch.gcu interfaces.")
    parser.add_argument(
        '--gen-rst', help="Generate RST documentation to the specified path.")
    parser.add_argument('--gen-excel',
                        help="Generate Excel file to the specified path.")
    parser.add_argument('--gen-csv',
                        help="Generate CSV file to the specified path.")
    parser.add_argument('--print-table',
                        action='store_true',
                        help="Print a pretty table to the terminal.")

    args = parser.parse_args()

    if torch.gcu is None:
        print("torch.gcu module not found.")
        return

    interface_mapping, unsupported_interfaces = compare_cuda_gcu_interfaces()

    if args.gen_rst:
        rst_content = generate_rst_from_mapping(interface_mapping,
                                                unsupported_interfaces)
        with open(args.gen_rst, 'w') as f:
            f.write(rst_content)
        print(f"RST content generated and saved to {args.gen_rst}")

    if args.gen_excel:
        generate_excel_from_mapping(interface_mapping, unsupported_interfaces,
                                    args.gen_excel)
        print(f"Excel content generated and saved to {args.gen_excel}")

    if args.gen_csv:
        generate_csv_from_mapping(interface_mapping, unsupported_interfaces,
                                  args.gen_csv)
        print(f"CSV content generated and saved to {args.gen_csv}")

    if args.print_table:
        print_pretty_table(interface_mapping, unsupported_interfaces)


if __name__ == "__main__":
    main()
