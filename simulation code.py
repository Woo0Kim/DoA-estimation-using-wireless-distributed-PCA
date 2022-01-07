import numpy as np
import time
import random

# Constants
NUMBER_OF_NODE = 20
NEIGHBOR_NODE = 8
NEIGHBOR_TABLE = []
MAX_AC_ITERATION = 20
DIVERSITY = 2
SNR = 0
TIME_SAMPLE = 7
POWER_METHOD_ITERATION = 5

class node:
    def __init__(self, value, node_number):
        global NUMBER_OF_NODE
        self._previous_value = value
        self._value = value
        self._is_updated = False
        self._node_num = node_number
        #connectivity
        self._neighbors = [(node_number+j)%NUMBER_OF_NODE for j in range(1, NEIGHBOR_NODE+1)]
        self._previous_c_ji = -1
        self._c_ji = -1
        self._u_ji = -1
        self._local_eigenvector = None

def Rayleigh(signal, diversity = DIVERSITY):
    # wired communication
    if diversity == 0: return signal
    # wireless communication
    else:
        h = (np.random.normal(0.0, 1.0, diversity)+1j*np.random.normal(0.0, 1.0,diversity))/np.sqrt(2)
        noise = (np.random.normal(0.0, 1.0,diversity)+1j*np.random.normal(0.0, 1.0,diversity))/np.sqrt(2)
        return (h*signal + noise, h)

def Hermitian(vec):
    return np.array(np.transpose(np.matrix(vec).getH()))[0]

def Demodulation(signal_tuple, diversity = DIVERSITY):
    # wired communication
    if diversity == 0: return signal_tuple
    # wireless communication
    else:
        signal = signal_tuple[0]
        h = signal_tuple[1]
        vec = Hermitian(h)/(np.linalg.norm(h)*np.linalg.norm(h))
        return np.dot(vec,signal)

def AC(j_node):
    global NEIGHBOR_TABLE, MAX_AC_ITERATION
    def AC_update_target_node(j_node):
        global NEIGHBOR_TABLE, TIME_SAMPLE
        if j_node._is_updated == True: return
        _sum = j_node._previous_value
        weight = 1/NEIGHBOR_NODE
        for node_number in j_node._neighbors:
            for i in range(TIME_SAMPLE):
                _sum[i] += weight * (Demodulation(Rayleigh(NEIGHBOR_TABLE[node_number]._previous_value[i])) - j_node._previous_value[i])
        j_node._value = _sum
        j_node._is_updated = True
        for node_number in j_node._neighbors:
            AC_update_target_node(NEIGHBOR_TABLE[node_number])

    # set a break time
    start_time = time.time()
    seconds = 0.5
    iteration = 0
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        AC_update_target_node(j_node)
        for node in NEIGHBOR_TABLE:
            node._previous_value = node._value
            node._is_updated = False
        iteration+=1
        if iteration >= MAX_AC_ITERATION: return
        elif elapsed_time > seconds: return

def create_steering_vector(NUMBER_OF_NODE, theta = np.pi/6, SENSOR_DIST = 100):
    steering_amp = 1
    W = 1000000000
    wave_length = 2 * np.pi * 300000000/W
    k= 2 * np.pi/wave_length
    steering_vector = np.empty(shape=(NUMBER_OF_NODE,), dtype= 'complex_')
    for index in range(NUMBER_OF_NODE):
        steering_vector[index] = np.exp(-1j*index*k*SENSOR_DIST*np.cos(theta))
    return steering_amp * steering_vector

def sourse_signal(t):
    global SNR
    def PHI(t): return np.random.normal(0,1)
    # SNR: 4dB
    A = 2.24137
    SNR = 10*np.log10(np.square(A)/2) 
    W = 1000000000
    return A * np.exp(W*t*1j+PHI(t))

def Initialization(NUMBER_OF_NODE, S):
    additive_noise = np.random.normal(0,1,NUMBER_OF_NODE)
    steering_vector = create_steering_vector(NUMBER_OF_NODE, theta=np.pi/6, SENSOR_DIST=100)
    received_signal = np.empty((0,NUMBER_OF_NODE))
    for time in range(S): received_signal = np.append(received_signal, np.array([steering_vector * sourse_signal(time) + additive_noise]), axis=0)
    node_data = np.transpose(received_signal)
    return node_data

def update_neighbor_table(NUMBER_OF_NODE, S, node_data):
    global NEIGHBOR_TABLE
    for i in range(NUMBER_OF_NODE):
        local_eigenvector = NUMBER_OF_NODE/S * np.dot(NEIGHBOR_TABLE[i]._value, node_data[i])
        local_eigenvector = local_eigenvector * np.ones(shape=(S,))
        NEIGHBOR_TABLE[i]._value = [x * v for x, v in zip(node_data[i], local_eigenvector)]
        NEIGHBOR_TABLE[i]._u_ji = local_eigenvector[0]

def Recursion(NUMBER_OF_NODE, S, node_number, node_data, global_eigenvector):
    global NEIGHBOR_TABLE, POWER_METHOD_ITERATION
    NEIGHBOR_TABLE = [node([x * v for x, v in zip(node_data[i], global_eigenvector)] , i) for i in range(NUMBER_OF_NODE)]
    for _ in range(POWER_METHOD_ITERATION):
        AC(NEIGHBOR_TABLE[node_number])
        update_neighbor_table(NUMBER_OF_NODE, S, node_data)

def Normalization(node_number):
    global NEIGHBOR_TABLE
    def Normalization_AC(j_node):
        global NEIGHBOR_TABLE, MAX_AC_ITERATION
        def Normalization_AC_update_target_node(j_node):
            global NEIGHBOR_TABLE
            if j_node._is_updated == True:
                return
            _sum = j_node._previous_c_ji
            weight = 1/NEIGHBOR_NODE
            for node_number in j_node._neighbors: 
                _sum += weight * (Demodulation(Rayleigh(NEIGHBOR_TABLE[node_number]._previous_c_ji)) - j_node._previous_c_ji)
            j_node._c_ji = _sum
            j_node._is_updated = True
            for node_number in j_node._neighbors: 
                Normalization_AC_update_target_node(NEIGHBOR_TABLE[node_number])

        # set a break time
        start_time = time.time()
        seconds = 0.5
        iteration = 0
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            Normalization_AC_update_target_node(j_node)
            for node in NEIGHBOR_TABLE:
                node._previous_c_ji = node._c_ji
                node._is_updated = False
            iteration+=1
            if iteration >= MAX_AC_ITERATION: return
            elif elapsed_time > seconds: return

    Normalization_AC(NEIGHBOR_TABLE[node_number])

def find_eigenvectors(NUMBER_OF_NODE,S):
    global NEIGHBOR_TABLE
    
    # Initialization
    node_data = Initialization(NUMBER_OF_NODE, S)
    eigen_vector = np.random.random(S)
    c_values = []
    u_values = []
    node_number = random.randrange(NUMBER_OF_NODE)
    
    # Distrubuted power method
    Recursion(NUMBER_OF_NODE, S, node_number, node_data, eigen_vector)
    for node_number in range(NUMBER_OF_NODE):
        NEIGHBOR_TABLE[node_number]._c_ji = NEIGHBOR_TABLE[node_number]._u_ji * np.conjugate(NEIGHBOR_TABLE[node_number]._u_ji)
        u_values.append(NEIGHBOR_TABLE[node_number]._u_ji)
        c_values.append(NEIGHBOR_TABLE[node_number]._c_ji)
        NEIGHBOR_TABLE[node_number]._previous_c_ji = NEIGHBOR_TABLE[node_number]._c_ji

    # Normalization
    Normalization(node_number)
    max_eigenvector = [(NEIGHBOR_TABLE[i]._u_ji / np.sqrt(NUMBER_OF_NODE*NEIGHBOR_TABLE[i]._c_ji)) for i in range(NUMBER_OF_NODE)]
    max_eigenvector = np.array(max_eigenvector)
    thetas = []
    del_theta = np.pi/36
    for i in range(1, 18):
        steering_vector = create_steering_vector(NUMBER_OF_NODE, theta=del_theta*i, SENSOR_DIST=100)
        thetas.append(np.absolute(np.dot(Hermitian(steering_vector), max_eigenvector)))
    return (np.argmax(thetas)+1)*del_theta

THETA  = np.pi/6
ERROR_SUM = 0
def simulation(iteration = 500):
    global THETA, ERROR_SUM
    data_for_variance = np.zeros(iteration)
    for i in range(iteration):
        estimation_error = abs(THETA - find_eigenvectors(NUMBER_OF_NODE, TIME_SAMPLE))
        data_for_variance[i] = estimation_error
        ERROR_SUM += estimation_error
        if i%10 == 0:
            print(i)
    print("Diversity:", DIVERSITY)
    print("SNR:", SNR)
    print('variance:', np.var(data_for_variance))
    print('Error:', ERROR_SUM/iteration)

simulation()