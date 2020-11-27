from torch import distributions as dist


def get_likelihood(str) -> dist:
    if str == 'bernoulli':
        return dist.Bernoulli
    elif str == 'categorical':
        return dist.OneHotCategorical
    elif str == 'laplace':
        return dist.Laplace
    elif str == 'normal':
        return dist.Normal
    else:
        print('likelihood not implemented')
        return None
