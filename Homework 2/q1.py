import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('data/images.csv')

df = df-df.mean()

cov_matrix = np.cov(pd.DataFrame(df).to_numpy(), rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_eigenvalues = eigenvalues.argsort()[::-1]
for x in sorted_eigenvalues[:10]:
    plt.imshow(np.reshape(eigenvectors[x], (48, 48)), cmap="gray")
    filename = "pca" + str(x) + ".jpg"
    plt.savefig(filename)
    plt.show()
    print("PCA with eigenvalue ", eigenvalues[x])
    print("proportion of variance explained: ", eigenvalues[x]/sum(eigenvalues))

x = [1, 10, 50, 100, 500]
y = []
for k in x:
    sum_eigenvalues = 0
    for i in range(k):
        sum_eigenvalues += eigenvalues[sorted_eigenvalues[i - 1]]
    ratio_variance = sum_eigenvalues / sum(eigenvalues)
    y.append(ratio_variance)

plt.plot(x, y)
plt.xlabel('k')
plt.ylabel('PVE')
plt.title('k vs PVE')
plt.show()


def reconstruction(k, eigenvectors, sorted_eigenvalues, image):
    eigenvectors_k = []
    for index in sorted_eigenvalues[:k]:
        eigenvector = eigenvectors[index]
        eigenvectors_k.append(eigenvector)
    eigenvectors_k = np.transpose(eigenvectors_k)
    score = np.dot(image, eigenvectors_k)
    recon = np.dot(score, np.transpose(eigenvectors_k))
    return recon

for k in [1, 10, 50, 100, 500]:
    recon = reconstruction(k, eigenvectors, sorted_eigenvalues, df.iloc[0])
    plt.imshow(np.reshape(recon, (48,48)), cmap="gray")
    picname = "reconstructed_" + str(k) + ".jpg"
    plt.savefig(picname)
    plt.show()



