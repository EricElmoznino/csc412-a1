import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import os


def sample_gaussian(n, d):
    return np.random.normal(loc=[0 for _ in range(d)], scale=[1 for _ in range(d)], size=(n, d))


def norm(x, keepdims=False):
    return np.linalg.norm(x, axis=-1, keepdims=keepdims)


def chi_pdf(k, g_of_x=lambda x: x, ginv_of_y=lambda y: y, dginv_of_y=lambda y: 1.0 * y):
    x = np.linspace(g_of_x(scipy.stats.chi.ppf(0.01, k)), g_of_x(scipy.stats.chi.ppf(0.99, k)), 100)
    y = dginv_of_y(scipy.stats.chi.pdf(ginv_of_y(x), k))
    return x, y


def alpha_mean_log_likelihood(xa, xb, polar, norms=False):
    alphas = np.linspace(0, 1, 100).reshape((-1, 1, 1))
    if not polar:
        interpolations = alphas * np.tile(xa, (alphas.shape[0], 1, 1)) + \
                         (1 - alphas) * np.tile(xb, (alphas.shape[0], 1, 1))
    else:
        interpolations = np.sqrt(alphas) * np.tile(xa, (alphas.shape[0], 1, 1)) + \
                         np.sqrt(1 - alphas) * np.tile(xb, (alphas.shape[0], 1, 1))
    if norms:
        interpolations = norm(interpolations, keepdims=True)
        log_likelihoods = scipy.stats.chi.logpdf(interpolations, xa.shape[-1])
    else:
        log_likelihoods = scipy.stats.norm.logpdf(interpolations)
    log_likelihoods = log_likelihoods.sum(axis=-1).mean(axis=-1)
    return alphas.reshape(-1), log_likelihoods.reshape(-1)


def normalized_hists(xs, legend):
    for x, l in zip(xs, legend):
        plt.hist(x, density=True, label=l, bins=25, alpha=0.3)


def line_plot(xs, legend):
    for (x, y), l in zip(xs, legend):
        plt.plot(x, y, label=l)


def save_plot(dir, title, x_label, y_label=''):
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(dir, title + '.png'), bbox_inches='tight')


plt.close()

# Part 2
x = sample_gaussian(n=1000, d=1)
x = norm(x)
normalized_hists([x], legend=['D=1'])
save_plot(dir='2', title='sample norms', x_label='norm(x)')

plt.close()

# Part 3/4
ds = [1, 2, 3, 10, 100]
xs = [sample_gaussian(n=1000, d=d) for d in ds]
xs = [norm(x) for x in xs]
normalized_hists(xs, legend=['D=%d hist' % d for d in ds])
save_plot(dir='3', title='sample norms', x_label='norm(x)')
xs = [chi_pdf(d) for d in ds]
line_plot(xs, legend=['D=%d pdf' % d for d in ds])
save_plot(dir='4', title='sample norms and pdf', x_label='norm(x) and x')

plt.close()

# Part 5
ds = [1, 2, 3, 10, 100]
xs = [sample_gaussian(n=1000, d=d) - sample_gaussian(n=1000, d=d) for d in ds]
xs = [norm(x) for x in xs]
normalized_hists(xs, legend=['D=%d hist' % d for d in ds])
xs = [chi_pdf(d,
              g_of_x=lambda x: np.sqrt(2) * x,
              ginv_of_y=lambda y: 1/np.sqrt(2) * y,
              dginv_of_y=lambda y: 1/np.sqrt(2) * y) for d in ds]
line_plot(xs, legend=['D=%d pdf' % d for d in ds])
save_plot(dir='5', title='normalized different between samples', x_label='norm(x)')

plt.close()

# Part 6/7
ds = [1, 2, 3, 10, 100]
xas, xbs = [sample_gaussian(n=1000, d=d) for d in ds], [sample_gaussian(n=1000, d=d) for d in ds]
likelihoods_linear = [alpha_mean_log_likelihood(xa, xb, polar=False) for xa, xb in zip(xas, xbs)]
likelihoods_polar = [alpha_mean_log_likelihood(xa, xb, polar=True) for xa, xb in zip(xas, xbs)]
for ll, lp, d in zip(likelihoods_linear, likelihoods_polar, ds):
    line_plot([ll], legend=['linear D=%d' % d])
    line_plot([lp], legend=['polar D=%d' % d])
    save_plot(dir='6', title='average log likelihoods D=%d' % d, x_label='alpha', y_label='log likelihood')
    plt.close()

plt.close()

# Part 8
ds = [1, 2, 3, 10, 100]
xas, xbs = [sample_gaussian(n=1000, d=d) for d in ds], [sample_gaussian(n=1000, d=d) for d in ds]
likelihoods_linear = [alpha_mean_log_likelihood(xa, xb, polar=False, norms=True) for xa, xb in zip(xas, xbs)]
likelihoods_polar = [alpha_mean_log_likelihood(xa, xb, polar=True, norms=True) for xa, xb in zip(xas, xbs)]
for ll, lp, d in zip(likelihoods_linear, likelihoods_polar, ds):
    line_plot([ll], legend=['linear D=%d' % d])
    line_plot([lp], legend=['polar D=%d' % d])
    save_plot(dir='8', title='average log likelihoods D=%d' % d, x_label='alpha', y_label='log likelihood')
    plt.close()

print('Done')
