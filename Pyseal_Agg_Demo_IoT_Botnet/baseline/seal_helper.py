# coding: utf-8
# author: Huelse

from seal import scheme_type


def print_example_banner(title):
    title_length = len(title)
    banner_length = title_length + 2 * 10
    banner_top = "+" + "-" * (banner_length - 2) + "+"
    banner_middle = "|" + ' ' * 9 + title + ' ' * 9 + "|"
    print(banner_top)
    print(banner_middle)
    print(banner_top)


def get_scheme_name(scheme):
    if scheme == scheme_type.BFV:
        scheme_name = "BFV"
    elif scheme == scheme_type.CKKS:
        scheme_name = "CKKS"
    else:
        scheme_name = "unsupported scheme"
    return scheme_name


def print_parameters(context):
    context_data = context.key_context_data()
    scheme_name = get_scheme_name(context_data.parms().scheme())
    print("/")
    print("| Encryption parameters:")
    print("| scheme: " + scheme_name)
    print("| poly_modulus_degree: " +
          str(context_data.parms().poly_modulus_degree()))
    print("| coeff_modulus size: ", end="")
    coeff_modulus = context_data.parms().coeff_modulus()
    coeff_modulus_sum = 0
    for j in coeff_modulus:
        coeff_modulus_sum += j.bit_count()
    print(str(coeff_modulus_sum) + "(", end="")
    for i in range(len(coeff_modulus) - 1):
        print(str(coeff_modulus[i].bit_count()) + " + ", end="")
    print(str(coeff_modulus[-1].bit_count()) + ") bits")
    if context_data.parms().scheme() == scheme_type.BFV:
        print("| plain_modulus: " +
              str(context_data.parms().plain_modulus().value()))
    print("\\")


def print_matrix(matrix, row_size):
    print()
    print_size = 5
    current_line = "    [ "
    for i in range(print_size):
        current_line += ((str)(matrix[i]) + ", ")
    current_line += ("..., ")
    for i in range(row_size - print_size, row_size):
        current_line += ((str)(matrix[i]))
        if i != row_size-1:
            current_line += ", "
        else:
            current_line += " ]"
    print(current_line)

    current_line = "    [ "
    for i in range(row_size, row_size + print_size):
        current_line += ((str)(matrix[i]) + ", ")
    current_line += ("..., ")
    for i in range(2*row_size - print_size, 2*row_size):
        current_line += ((str)(matrix[i]))
        if i != 2*row_size-1:
            current_line += ", "
        else:
            current_line += " ]"
    print(current_line)
    print()


def print_vector(vec, print_size=4, prec=3):
    slot_count = len(vec)
    print()
    if slot_count <= 2*print_size:
        print("    [", end="")
        for i in range(slot_count):
            print(" " + (f"%.{prec}f" % vec[i]) + ("," if (i != slot_count - 1) else " ]\n"), end="")
    else:
        print("    [", end="")
        for i in range(print_size):
            print(" " + (f"%.{prec}f" % vec[i]) + ",", end="")
        if len(vec) > 2*print_size:
            print(" ...,", end="")
        for i in range(slot_count - print_size, slot_count):
            print(" " + (f"%.{prec}f" % vec[i]) + ("," if (i != slot_count - 1) else " ]\n"), end="")
    print()
