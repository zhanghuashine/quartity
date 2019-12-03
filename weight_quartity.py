def weight_quantize(index, activation):

    tensor_distubution = np.zeros(INTERVAL_NUM)
    print("\nQuantize the Layer{}_Weight:".format(index))

    max_val = np.max(activation)
    min_val = np.min(activation)
    tensor_max = max(abs(max_val), abs(min_val))
    distubution_interval = STATISTIC * tensor_max / INTERVAL_NUM

    th = tensor_max
    hist, hist_edge = np.histogram(activation, bins=INTERVAL_NUM, range=(0, th))  # ?
    tensor_distubution += hist

    distribution = np.array(tensor_distubution)

    threshold_bin = threshold_distribution(distribution)  # 作用？（比较耗时）

    threshold = (threshold_bin + 0.5) * distubution_interval

    activation_scale = QUANTIZE_NUM / threshold
    # print("%-8d threshold : %-10f interval : %-10f scale : %-10f" % (
    # threshold_bin, threshold, distubution_interval, activation_scale))

    return activation_scale


def threshold_distribution(distribution, target_bin=128):
    distribution = distribution[1:]
    length = distribution.size
    threshold_sum = sum(distribution[target_bin:])
    kl_divergence = np.zeros(length - target_bin)

    for threshold in range(target_bin, length):
        sliced_nd_hist = copy.deepcopy(distribution[:threshold])

        p = sliced_nd_hist.copy()
        p[threshold - 1] += threshold_sum
        threshold_sum = threshold_sum - distribution[threshold]

        is_nonzeros = (p != 0).astype(np.int64)

        quantized_bins = np.zeros(target_bin, dtype=np.int64)

        num_merged_bins = sliced_nd_hist.size // target_bin

        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()

        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[p == 0] = 0
        p[p == 0] = 0.0001
        q[q == 0] = 0.0001

        kl_divergence[threshold - target_bin] = stats.entropy(p, q)

    min_kl_divergence = np.argmin(kl_divergence)
    threshold_value = min_kl_divergence + target_bin

    return threshold_value
