# Small test script for basic bugs
data = np.random.normal(10, 69, (10,10,5))
print(data.shape)
k = 4
connect = Connectivity(data, k)
print(connect.W.shape)
print(connect.G.shape)
print(connect.sigmas.shape)

LL = connect.log_likelihood()
print(LL)

LLs = connect.fit()
print(len(LLs))
plt.plot(LLs)
