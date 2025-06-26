from numpy import *

def compute_error_for_line(b, m, points):
    tot_error = 0
    for i in range(0, len(points)):
         x = points[i][0]
         y = points[i][1]
         tot_error += (y - (m*x + b))**2
    return tot_error/ float(len(points))

def gradient_descent_step(b_curr, m_curr, points, lr):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        b_gradient += -(2/N) * (y-(m_curr * x) + b_curr)
        m_gradient += -(2/N) * x * (y-(m_curr * x) + b_curr)
        
    new_b = b_curr - (lr * b_gradient)
    new_m = m_curr - (lr * m_gradient)
    
    return [new_b, new_m]


def gradient_descent_runner(points, start_b, start_m, lr, num_iter):
    b = start_b
    m = start_m
    for i in range(num_iter):
        b, m = gradient_descent_step(b, m, array(points), lr)

    return [b, m]
         

def run():
    # Collect Data
    points = genfromtxt('./data.csv', delimiter=',')
    print(points[0])
    # Define Hyperparameters
    # y= mx+b
    learning_rate = 0.0001
    initial_m = 0
    initial_b = 0
    num_iterations = 1000
    
    # Train Model
    print(f"Starting Gradient descent at b = {initial_b}, m={initial_m}, error= {compute_error_for_line(initial_b, initial_m, points)}")
    
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    
    print(f"Ending Gradient descent at b = {b}, m={m}, error= {compute_error_for_line(b, m, points)}")
    
    
if __name__ == "__main__":
    run()
