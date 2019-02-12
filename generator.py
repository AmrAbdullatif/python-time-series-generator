def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
 """
    Perform Python generator that takes the current array of time series data and yields 
    batches of data from the recent past, along with a target. It can be used for MIMO, SISO, MISO, and SIMO.

    Parameters
    ----------
    data : numpy.ndarray
        2d tensor of shape (timesteps, input_features)
    lookback: int
        How many timesteps back the input data should go.
    delay: int
        How many timesteps in the future the target should be.
    min_index and max_index: int
        Indices in the data array that delimit which timesteps to draw from.
    shuffle: boolean
        Whether to shuffle the samples or draw them in chronological order.
    batch_size: int
        The number of samples per batch.
    step: int
        The period, in timesteps, at which you sample data.
    
    Returns
    -------
    samples : 3d tensor
        timeseries samples (batch_size, timesteps_lookback, number_inputs_features)
    targets: 2d tensor
        timeseries targets (batch_size, number_targets)
    """
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index + 1, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index  + 1))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),data.shape[-1]))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay]
        yield samples, targets

Example:

data_ex = np.random.random((100,3))

lookback = 10
step = 1
delay = 1
batch_size = 128

train_gen = generator(xp,
    lookback=lookback,
    delay=delay,
    min_index=0,
    max_index=50,
    shuffle=True,
    step=step,
    batch_size=batch_size)

validation_gen = generator(xp,
    lookback=lookback,
    delay=delay,
    min_index=51,
    max_index=75,
    step=step,
    batch_size=batch_size)

test_gen = generator(xp,
    lookback=lookback,
    delay=delay,
    min_index=76,
    max_index=None,
    step=step,
    batch_size=batch_size)
    
# Access the data
xi,yo = next(train_gen)
# xi shape is (128, 10, 3)  
# yo shape is (128,3) 


