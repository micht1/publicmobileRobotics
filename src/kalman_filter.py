# initial_kalman_values = X(0) inital state
# initial_input = u(0) initial input
# initial_covariance = intial covarience
# measured_states = measurements of the camera and motor speeds
# A , B and C model of the system
# Q, R covariences of states and measurements
# https://en.wikipedia.org/wiki/Kalman_filter

def kalman_filt(initial_kalman_values,initial_input,initial_covariance,measured_states,A,B,C,Q,R):
    Q=np.array(Q)
    predict_position=np.matmul(A.astype(float),ini_pose.astype(float))+np.matmul(B.astype(float),initial_input.astype(float))
    predict_cov=np.matmul(np.matmul(A.astype(float),initial_covariance.astype(float)),np.transpose(A))+Q
    i=measured_states-np.matmul(C.astype(float),predict_position.astype(float))
    S=np.matmul(np.matmul(C.astype(float),predict_cov.astype(float)),np.transpose(C))+R
    if S.shape==(1,):
        K=predict_cov*np.transpose(C)*1/(S)
    else:
        K=np.matmul(np.matmul(predict_cov.astype(float),np.transpose(C.astype(float))),np.linalg.inv(S.astype(float)))
    thymio_position=predict_position+np.matmul(K.astype(float),i.astype(float))
    if Q.shape==():
        cov=(1-K*C)*predict_cov
    else:
        cov=np.matmul((np.eye(len(Q.astype(float)))-np.matmul(K.astype(float),C.astype(float))),predict_cov)

    return(thymio_position,cov)
