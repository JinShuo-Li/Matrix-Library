import copy

#From ID generate sequence
def generate_sequence_from_id(student_id):
    if int(student_id) == 0:
        return [0]*289
    sequence = []
    digits = [int(char) for char in student_id]
    power = 1
    while len(sequence) < 289:
        for digit in digits:
            power_result = digit ** power
            for char in str(power_result):
                sequence.append(int(char))
                if len(sequence) >= 289:
                    return sequence
                else:
                    continue
        power += 1

#Slice the sequence into matrix
def generate_matrix(seq):
    mat = []
    for i in range(17):
        row = []
        for j in range(17):
            ele = i*17 + j
            row.append(seq[ele])
        mat.append(row)
    return mat

#Matrix transpose
def transpose(mat):
    if not mat or not mat[0]:
        return []
    
    rows = len(mat)
    cols = len(mat[0])
    
    transposed = []
    for j in range(cols):
        new_row = []
        for i in range(rows):
            if i < len(mat) and j < len(mat[i]):
                new_row.append(mat[i][j])
            else:
                new_row.append(0)
        transposed.append(new_row)
    return transposed

#Transfer the matrix into Row Echelon Form(REF)
def gaussian_elimination(mat):
    matrix = copy.deepcopy(mat)
    n = len(matrix)
    m = len(matrix[0])
    index = 0
    counter = 0
    for j in range(m):
        if index >= n:
            break
        t = index
        while t < n and matrix[t][j] == 0:
            t += 1
        if t == n:
            continue
        else:
            if index != t:
                counter += 1
            matrix[index], matrix[t] = matrix[t], matrix[index]
            factor = matrix[index][j]
            for row in range(index+1,n):
                if matrix[row][j] == 0:
                    continue
                else:
                    multiplier = matrix[row][j] / factor
                    for col in range(j,m):
                        matrix[row][col] = matrix[row][col] - multiplier * matrix[index][col]
        index += 1
    return matrix, counter

#Transfer the matrix into Row Reduced Echelon Form(RREF)
def gaussian_elimination_simplified(mat):
    matrix = copy.deepcopy(mat)
    n = len(matrix)
    m = len(matrix[0])
    index = 0
    counter = 0
    
    for j in range(m):
        if index >= n:
            break
        t = index
        while t < n and matrix[t][j] == 0:
            t += 1
        if t == n:
            continue
        else:
            if index != t:
                counter += 1
            matrix[index], matrix[t] = matrix[t], matrix[index]
            
            pivot = matrix[index][j]
            if pivot != 0:
                for col in range(j, m):
                    matrix[index][col] = matrix[index][col] / pivot
            
            for row in range(n):
                if row == index:
                    continue
                if matrix[row][j] != 0:
                    multiplier = matrix[row][j]
                    for col in range(j, m):
                        matrix[row][col] = matrix[row][col] - multiplier * matrix[index][col]
            
            index += 1
    return matrix, counter

#Determine the rank of a matrix
def rank_of_matrix(mat):
    matrix = [row[:] for row in mat]
    n, m = len(matrix), len(matrix[0])
    rank = 0
    for j in range(m):
        pivot = None
        for i in range(rank, n):
            if matrix[i][j] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]

        for i in range(rank + 1, n):
            if matrix[i][j] != 0:
                ratio = matrix[i][j] / matrix[rank][j]
                for col in range(j, m):
                    matrix[i][col] -= ratio * matrix[rank][col]
        rank += 1
    return rank

#Format the matrix
def format_matrix(matrix, decimals=3):
    formatted = []
    for row in matrix:
        formatted_row = []
        for value in row:
            if abs(value) < 1e-10:
                formatted_value = 0.0
            else:
                formatted_value = round(value, decimals)
            formatted_row.append(formatted_value)
        formatted.append(formatted_row)
    return formatted

#Calculate the determinant using REF/RREF(To use this function, please in put a REF and a index)
def det_REF(mat, counter):
    m = len(mat)
    n = len(mat[0])
    result = 1
    if m != n:
        print("False")
        return None
    else:
        for i in range(m):
            result *= mat[i][i]
    result = result * ((-1)**(counter))
    return result

#Calculate the determinant using Laplace method and Recursion
def det(mat):
    result = 0
    m = len(mat)
    n = len(mat)
    if m != n or m == 0:
        return None
    elif m == 1:
        return mat[0][0]
    elif m == 2:
        return (mat[0][0] * mat[1][1]) - (mat[1][0] * mat[0][1])
    else:
        for j in range(m):
            new_mat = []
            if mat[0][j] == 0:
                continue
            for i in range(1,m):
                row = []
                for k in range(m):
                    if j == k:
                        continue
                    else:
                        row.append(mat[i][k])
                new_mat.append(row)
            result += ((-1)**(j))*(det(new_mat))*(mat[0][j])
    return result

#Full rank decomposition
def full_rank_decomposition(matrix):
    r = rank_of_matrix(matrix)
    rref_matrix, _ = gaussian_elimination_simplified(matrix)
    pivot_columns = []
    rows = len(rref_matrix)
    cols = len(rref_matrix[0]) if rows > 0 else 0
    for i in range(min(r, rows)):
        for j in range(cols):
            if abs(rref_matrix[i][j]) > 1e-10:
                pivot_columns.append(j)
                break
    F = []
    for i in range(len(matrix)):
        new_row = []
        for j in pivot_columns:
            new_row.append(matrix[i][j])
        F.append(new_row)
    G = rref_matrix[:r]
    return F, G, r

#Find solutions for homogeneous system
def solve_homogeneous_system(A):
    if not A or not A[0]:
        return []
    
    R,_ = gaussian_elimination_simplified(A)
    n_rows = len(R)
    n_cols = len(R[0]) if n_rows > 0 else 0
    
    pivot_columns = []
    for i in range(n_rows):
        for j in range(n_cols):
            if abs(R[i][j]) > 1e-10:
                pivot_columns.append(j)
                break
    
    free_columns = [j for j in range(n_cols) if j not in pivot_columns]
    
    if not free_columns:
        return []
    
    basis_vectors = []
    for free_idx in free_columns:
        x = [0.0] * n_cols
        x[free_idx] = 1.0
        
        for i in range(n_rows-1, -1, -1):
            pivot_col = -1
            for j in range(n_cols):
                if abs(R[i][j]) > 1e-10:
                    pivot_col = j
                    break
            if pivot_col == -1:
                continue
                
            s = 0.0
            for j in range(pivot_col+1, n_cols):
                s += R[i][j] * x[j]
            x[pivot_col] = -s / R[i][pivot_col]
        
        basis_vectors.append(x)
    
    return basis_vectors

#Find maximal linearly independent subset
def find_maximal_linearly_independent_subset(mat):
    matrix = transpose(mat)
    result_subset = []
    test_subset = []
    index = 0
    for i in matrix:
        test_subset.append(i)
        if rank_of_matrix(test_subset) == index + 1:
            result_subset.append(i)
            index += 1
        else:
            test_subset.pop()
    return transpose(result_subset)

#Matrix basic operation: addition
def matrix_addition(mat_1,mat_2):
    if len(mat_1) != 0 and len(mat_2) != 0:
        if len(mat_1[0]) != 0 and len(mat_2[0]) != 0:
            pass
        else:
            print("Empty Matrix")
            return None
    else:
        print("Empty Matrix")
        return None
    result = []
    if len(mat_1) == len(mat_2) and len(mat_1[0]) == len(mat_2[0]):
        for i in range(len(mat_1)):
            row = []
            for j in range(len(mat_1[0])):
                new_element = mat_1[i][j] + mat_2[i][j]
                row.append(new_element)
            result.append(row)
        return result
    else:
        print("Format not matched")
        return None

#Vector basic operation: dot production
def vector_dot_production(vec_1, vec_2):
    if len(vec_1) != 0 and len(vec_2) != 0 and len(vec_1) == len(vec_2):
        ans = 0
        for i in range(len(vec_1)):
            ans += vec_1[i]*vec_2[i]
        return ans
    else:
        print("Error")
        return None

#Vector basic operation: Substitution
def vector_substitution(vec_1, vec_2):
    if len(vec_1) != 0 and len(vec_2) != 0 and len(vec_1) == len(vec_2):
        vec_res = []
        for i in range(len(vec_1)):
            vec_res.append(vec_1[i]-vec_2[i])
        return vec_res
    else:
        print("Error")
        return None

#Vector basic operation: Scalar multiplication
def vector_scalar_product(k,vec):
    if len(vec) != 0:
        result = []
        for i in vec:
            result.append(k*i)
        return result
    else:
        print("Error")
        return None

#Schimidt Orthogonalization
def Schimidt_Orthogonalization(mat):
    if len(mat) == 0 or len(mat[0]) == 0:
        print("Empty Matrix")
        return None
    else:
        matrix = transpose(mat)
        result = []
        for i in range(len(matrix)):
            row = []
            assist = matrix[i]
            if i == 0:
                row = matrix[i]
            else:
                for j in range(i):
                    factor = vector_dot_production(matrix[i],result[j])/vector_dot_production(result[j],result[j])
                    assist = vector_substitution(assist, vector_scalar_product(factor,result[j]))
                row = assist
            result.append(row)
        res = transpose(result)
        return res

#Congruent Diagonization
def congruent_diagonalization(mat):
    matrix = copy.deepcopy(mat)
    n = len(matrix)

    for k in range(n):
        pivot_row = k
        while pivot_row < n and matrix[pivot_row][k] == 0:
            pivot_row += 1
        if pivot_row == n:
            continue
        if pivot_row != k:
            matrix[k], matrix[pivot_row] = matrix[pivot_row], matrix[k]
            for i in range(n):
                matrix[i][k], matrix[i][pivot_row] = matrix[i][pivot_row], matrix[i][k]
        
        pivot = matrix[k][k]
        if pivot == 0:
            continue
        for i in range(k+1, n):
            if matrix[i][k] != 0:
                factor = matrix[i][k] / pivot
                for j in range(k, n):
                    matrix[i][j] = matrix[i][j] - factor * matrix[k][j]
        
        for j in range(k+1, n):
            if matrix[k][j] != 0:
                factor = matrix[k][j] / pivot
                for i in range(k, n):
                    matrix[i][j] = matrix[i][j] - factor * matrix[i][k]
    return matrix

#Calculate the indertia index
def find_inertia_index(diag_mat):
    seq = []
    for i in range(len(diag_mat)):
        seq.append(diag_mat[i][i])
    inertia_positive = 0
    inertia_negative = 0
    inertia_zero = 0
    for i in seq:
        if abs(i) < 1e-10:
            inertia_zero += 1
        elif i > 0:
            inertia_positive += 1
        else:
            inertia_negative += 1
    return inertia_positive, inertia_negative, inertia_zero

#Input
stu_id = input("Please input your id(No special sign, no spaces):")

#Calculations
sequence = generate_sequence_from_id(stu_id)
mat = generate_matrix(sequence)
REF_ori, counter = gaussian_elimination(mat)
determinant = det_REF(REF_ori,counter)
RREF_ori, _ = gaussian_elimination_simplified(mat)
basis = solve_homogeneous_system(mat)
full_col_rank_mat, full_row_rank_mat, _ = full_rank_decomposition(mat)
max_linearly_independent_subset = find_maximal_linearly_independent_subset(mat)
orthogonalized_vector_set = Schimidt_Orthogonalization(max_linearly_independent_subset)
matrix = matrix_addition(mat,transpose(mat))
congruent_diag = congruent_diagonalization(matrix)
p, n, z = find_inertia_index(congruent_diag)

def dot(a,b):
    res = 0
    for i in range(len(a)):
        res += a[i]*b[i]
    return res

for i in range(len(orthogonalized_vector_set)):
    for j in range(i,len(orthogonalized_vector_set)):
        if i == j:
            continue
        else:
            dp = dot(orthogonalized_vector_set[i],orthogonalized_vector_set[j])
            if abs(dp) > 1e-10:
                print("Error in orthogonalization")
#Outputs
print(f"The result is shown below")
print(f"The determinant |A| = {determinant}.")
print(f"The RREF is:")
for row in format_matrix(RREF_ori,2):
    print(row)
if len(basis) == 0:
    print(f"The equation Ax=0 only has the zero solution.")
else:
    print(f"The general solution is: x = ")
    format_basis = format_matrix(basis)
    for i in range(len(basis)):
        if i+1 < len(basis):
            print(f"{format_basis[i]}c{i+1} + ")
        else:
            print(f"{format_basis[i]}c{i+1}")
    print(f"c1 ~ c{len(basis)} are arbitrary constants.")
print(f"The full colume rank matrix is:")
for row in format_matrix(full_col_rank_mat, 2):
    print(row)
print(f"The full row rank matrix is:")
for row in format_matrix(full_row_rank_mat):
    print(row)
print(f"The Max linearly independent subset is:")
for row in max_linearly_independent_subset:
    print(row)
print(f"The orthogonalized vector set is:")
for row in format_matrix(orthogonalized_vector_set,2):
    print(row)
print(f"The positive inertial index is {p}")
print(f"The negative inertial index is {n}")

input("Press ENTER to exit.")