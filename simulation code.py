# Object: Find out if the result match Fig3 of paper.
# (result from the number of neighbors per node ranging from 2 to 16.)

# parameters: NUMBER_OF_NODE = 20, ac_interation:k = 20, power_method_interation: n = 5
# NUMBER_OF_NODE: the number of sensors, n: the recursion of the power method, k: iterations of the AC protocol
import numpy as np
import time
import matplotlib.pyplot as plt

################ plot 해야할 목록 
# 1. 기존 기법과 채널 들어간거 비교(채널 없을 때, 레일리 페이딩 채널 div=1, div=2, div=3) (x축: # of Neighbors, y축: avg estimation error)
# 조건: 노드수 20, power method recursion: 5, AC algorithm iteration: 20, SNR = 0db, 5db, 20db일 때. (A와 W도 명시)
# 2. SNR에 따른 estimation error?? 근데 이거 1번에 포함되어 있는거 아닌가?
# 3

# const values
# neighboring node per node = 7
NUMBER_OF_NODE = 20
NEIGHBOR_NODE = 8
NEIGHBOR_TABLE = []
MAX_AC_ITERATION = 20
DIVERSITY = 2
SNR = 0
TIME_SAMPLE = 7


class node:
    def __init__(self, value, node_number):
        global NUMBER_OF_NODE

        self._previous_value = value
        self._value = value
        self._is_updated = False
        self._node_num = node_number
        self._neighbors = [(node_number+j)%NUMBER_OF_NODE for j in range(1, NEIGHBOR_NODE+1)]
        self._previous_c_ji = -1
        self._c_ji = -1
        self._u_ji = -1

#DIVERSITY가 0 이면 그냥 유선통신(레일리 적용 X)
def Rayleigh(signal, diversity = DIVERSITY):
    if diversity == 0:
        return signal
    else:
        h = (np.random.normal(0.0, 1.0, diversity)+1j*np.random.normal(0.0, 1.0,diversity))/np.sqrt(2)
        noise = (np.random.normal(0.0, 1.0,diversity)+1j*np.random.normal(0.0, 1.0,diversity))/np.sqrt(2)
        return (h*signal + noise, h)


def Demodulation(signal_tuple, diversity = DIVERSITY):
    if diversity == 0:
        return signal_tuple
    else:
        signal = signal_tuple[0]
        h = signal_tuple[1]
        vec = Hermitian(h)/(np.linalg.norm(h)*np.linalg.norm(h))
        return np.dot(vec,signal)

def Hermitian(vec):
    return np.array(np.transpose(np.matrix(vec).getH()))[0]

def AC(j_node):
    global NEIGHBOR_TABLE, MAX_AC_ITERATION
    def AC_update_target_node(j_node):
        global NEIGHBOR_TABLE

        if j_node._is_updated == True: 
            return
        
        _sum = j_node._previous_value
        weight = 1/NEIGHBOR_NODE
        for node_number in j_node._neighbors:
            _sum += weight * (Demodulation(Rayleigh(NEIGHBOR_TABLE[node_number]._previous_value)) - j_node._previous_value)
        j_node._value = _sum
        j_node._is_updated = True

        for node_number in j_node._neighbors:
            AC_update_target_node(NEIGHBOR_TABLE[node_number])


    #Initialization Phase
    start_time = time.time()
    seconds = 0.5
    
    #Clock Fired
    iteration = 0
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time

        AC_update_target_node(j_node)

        for node in NEIGHBOR_TABLE:
            node._previous_value = node._value
            node._is_updated = False

        iteration+=1

        if iteration >= MAX_AC_ITERATION:
            return j_node._value
        elif elapsed_time > seconds: 
            return j_node._value



# sensor_dist = 1m, wave_length = source signal에 맞춰서
def create_steering_vector(NUMBER_OF_NODE, theta = np.pi/6, SENSOR_DIST = 100):
    steering_amp = 1
    W = 1000000000 #10^9 radian per second
    wave_length = 2 * np.pi * 300000000/W #300000km
    k= 2 * np.pi/wave_length
    steering_vector = np.empty(shape=(NUMBER_OF_NODE,), dtype= 'complex_')
    for index in range(NUMBER_OF_NODE):
        steering_vector[index] = np.exp(-1j*index*k*SENSOR_DIST*np.cos(theta))
    
    return steering_amp * steering_vector


def sourse_signal(t):
    global SNR
    def PHI(t):
        return np.random.normal(0,1)

    A = 3.55234 # amplitude can be change 14.2, 2.4, sqrt(2)+0.02
    SNR = 10*np.log10(np.square(A)/2)
    W = 1000000000 #10^9 radian per second
    return A * np.exp(W*t*1j+PHI(t))

def Initialization(NUMBER_OF_NODE, S):
    # additive_noise = 0
    additive_noise = np.random.normal(0,1,NUMBER_OF_NODE)
    steering_vector = create_steering_vector(NUMBER_OF_NODE, theta=np.pi/6, SENSOR_DIST=100)
    received_signal = np.empty((0,NUMBER_OF_NODE))

    for time in range(S):
        #received_signal = np.append(received_signal, np.array([steering_vector * sourse_signal(time)]), axis=0)
        received_signal = np.append(received_signal, np.array([steering_vector * sourse_signal(time) + additive_noise]), axis=0)

    node_data = np.transpose(received_signal)

    return node_data


def Recursion(NUMBER_OF_NODE, S, node_number, node_data, eigen_vector):
    #Create NEIGHBOR_TABLE
    global NEIGHBOR_TABLE
    NEIGHBOR_TABLE = [node(np.dot(Hermitian(node_data[i]), eigen_vector), i) for i in range(NUMBER_OF_NODE)]

    #Recursion(각 노드에서 이걸 돌린다고 생각해야할듯).
    for _ in range(5):
        eigen_vector = NUMBER_OF_NODE*AC(NEIGHBOR_TABLE[node_number])
        eigen_vector = (eigen_vector*sum(node_data[node_number])/S) * np.ones(shape=(S,))

    return eigen_vector[0]

def Normalization(NUMBER_OF_NODE, node_number):
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


        #Initialization Phase
        start_time = time.time()
        seconds = 0.5
        
        iteration = 0
        #Clock Fired
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            Normalization_AC_update_target_node(j_node)

            for node in NEIGHBOR_TABLE:
                node._previous_c_ji = node._c_ji
                node._is_updated = False

            iteration+=1

            if iteration >= MAX_AC_ITERATION:
                return j_node._c_ji
            elif elapsed_time > seconds: 
                return j_node._c_ji

    #Normalization(이것도 일단 작성한 다음에 각 노드에서 돌린다고 생각하자 - c값 받고 각 노드에 저장한 다음에 돌리기? 일단 모든 노드에는 recursion에서 각자 계산한 c 값들이 존재하는 듯)
    c= np.sqrt(NUMBER_OF_NODE * Normalization_AC(NEIGHBOR_TABLE[node_number]))
    return NEIGHBOR_TABLE[node_number]._u_ji/c


def find_eigenvectors(NUMBER_OF_NODE,S):
    global NEIGHBOR_TABLE
    
    # Initialization
    node_data = Initialization(NUMBER_OF_NODE, S)
    eigen_vector = np.random.random(S)

    # Recursion (이거 지금 각 노드에서 초기화 시키고 처음부터 돌리는 형식이라 비효율적인데 나중에 고민해보고 수정해보자)
    max_eigenvector = []
    #TODO 이거도 비효율적
    c_values = []
    u_values = []
    for node_number in range(NUMBER_OF_NODE):
        NEIGHBOR_TABLE[node_number]._u_ji = Recursion(NUMBER_OF_NODE, S, node_number, node_data, eigen_vector)
        NEIGHBOR_TABLE[node_number]._c_ji = NEIGHBOR_TABLE[node_number]._u_ji * np.conjugate(NEIGHBOR_TABLE[node_number]._u_ji)
        u_values.append(NEIGHBOR_TABLE[node_number]._u_ji)
        c_values.append(NEIGHBOR_TABLE[node_number]._c_ji)
        NEIGHBOR_TABLE[node_number]._previous_c_ji = NEIGHBOR_TABLE[node_number]._c_ji

    # Normalization (이거 지금 각 노드에서 초기화 시키고 처음부터 돌리는 형식이라 비효율적인데 나중에 고민해보고 수정해보자)
    for node_number in range(NUMBER_OF_NODE):
        #이거 고쳐야하는듯
        for idx, val in enumerate(c_values):
            NEIGHBOR_TABLE[idx]._u_ji = u_values[idx]
            NEIGHBOR_TABLE[idx]._c_ji = val
            NEIGHBOR_TABLE[idx]._previous_c_ji = val
        max_eigenvector.append(Normalization(NUMBER_OF_NODE, node_number))

    
    max_eigenvector = np.array(max_eigenvector)
    thetas = []
    del_theta = np.pi/36
    for i in range(1, 18):
        steering_vector = create_steering_vector(NUMBER_OF_NODE, theta=del_theta*i, SENSOR_DIST=100)
        thetas.append(np.absolute(np.dot(Hermitian(steering_vector), max_eigenvector)))
        
        #print(i, np.absolute(np.dot(Hermitian(steering_vector), max_eigenvector)))
    #print(np.argmax(thetas)+1)
    # print(123, np.rad2deg((np.argmax(thetas)+1)*del_theta))
    # print(1234, np.argmax(thetas))
    return (np.argmax(thetas)+1)*del_theta

THETA  = np.pi/6
ERROR_SUM = 0
def simulation(iteration = 500):
    global THETA, ERROR_SUM
    data_for_variance = np.zeros(iteration)
    for i in range(iteration):
        # print(321, np.rad2deg(THETA))
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
