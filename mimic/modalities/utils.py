from torch import distributions as dist


def get_likelihood(str):
    if str == 'laplace':
        pz = dist.Laplace;
    elif str == 'bernoulli':
        pz = dist.Bernoulli;
    elif str == 'normal':
        pz = dist.Normal;
    elif str == 'categorical':
        pz = dist.OneHotCategorical;
    else:
        print('likelihood not implemented')
        pz = None;
    return pz;
