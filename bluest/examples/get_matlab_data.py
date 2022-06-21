def save_matlab_data(filename, sap):
    from scipy.io import savemat

    #var = (psi@m).reshape((N,N))

    data = sap.__dict__.copy()
    keys = [key for key,value in data.items() if value is None or callable(value)]
    for key in keys:
        data.pop(key)

    savemat(filename, data)
