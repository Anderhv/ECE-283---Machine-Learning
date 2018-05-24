import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.executing_eagerly()

### 1) Create synthetic data
# Class 0
mu_0 = [1, 1]
cov_0 = [[1, 0], [0, 1]]
x_0, y_0 = np.random.multivariate_normal(mu_0, cov_0, 200).T

# Class 1
#mu_1 = [1, 1]
#cov_1 = [[1, 0], [0, 1]]
#x_1, y_1 = np.random.multivariate_normal(mu_1, cov_1, 200).T

pi_A = 1/3;
pi_B = 2/3;


tfd = tf.contrib.distributions
bimix_gauss = tfd.Mixture(
  cat=tfd.Categorical(probs=[pi_A, pi_B]),
  components=[
    tfd.Normal(loc=-1., scale=0.1),
    tfd.Normal(loc=+1., scale=0.5),
])


x = tf.linspace(-2., 3., int(1e4)).eval()
plt.plot(x, bimix_gauss.prob(x).eval());


#plt.plot(x_0, y_0, 'x')
plt.show()

