import numpy as np

# Define the geometry and boundary conditions
vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
elements = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
dirichlet_bc = np.array([1, 0, 0, 0])
num_vertices = vertices.shape[0]

# Define the Laplace equation and its derivative
def laplace_single_layer(x, y, x0, y0):
    return -1/(2*np.pi)*np.log(np.sqrt((x-x0)**2 + (y-y0)**2))

def laplace_double_layer(x, y, x0, y0):
    return 1/(2*np.pi)*((x-x0)*(x-x0) + (y-y0)*(y-y0))/((x-x0)**2 + (y-y0)**2)

A = np.zeros((num_vertices, num_vertices))
b = dirichlet_bc.copy()

for i in range(elements.shape[0]):
    x1, y1 = vertices[elements[i, 0]]
    x2, y2 = vertices[elements[i, 1]]
    n1, n2 = (y2-y1), -(x2-x1)
    length = np.sqrt(n1**2 + n2**2)
    n1 /= length
    n2 /= length

    for j in range(num_vertices):
        if j == elements[i, 0]:
            A[j, j] += 1/2
        elif j == elements[i, 1]:
            A[j, j] += 1/2
        else:
            x0, y0 = vertices[j]
            A[j, elements[i, 0]] += laplace_single_layer(x1, y1, x0, y0)*n1 - laplace_double_layer(x1, y1, x0, y0)*n2
            A[j, elements[i, 1]] += laplace_single_layer(x2, y2, x0, y0)*n1 - laplace_double_layer(x2, y2, x0, y0)*n2
phi = np.linalg.solve(A, b)
x, y = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
u = np.zeros_like(x)

for i in range(x.size):
    for j in range(num_vertices):
        u.flat[i] += laplace_single_layer(x.flat[i], y.flat[i], vertices[j, 0], vertices[j, 1])*phi[j]

# Print the estimated result
print(u)
