import numpy as np
import torch

from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import falkon.hopt
from falkon import FalkonOptions
from falkon.hopt.objectives import SimpleVal

np.random.seed(777777)

# An L1 loss function
def mclass_loss(true, pred):
    mae = torch.nn.L1Loss()
    return mae(true, pred)

# Here we select the CV subspace
subspace = "sumnc"

# Then we load the associated datasets
X_ref = np.loadtxt("datasets/"+subspace+"/X_ref.txt")
X_train = np.loadtxt("datasets/"+subspace+"/X_train.txt")
X_test = np.loadtxt("datasets/"+subspace+"/X_test.txt")

Y_ref = np.loadtxt("datasets/"+subspace+"/p_ref.txt")
Y_train = np.loadtxt("datasets/"+subspace+"/p_train.txt")
Y_test = np.loadtxt("datasets/"+subspace+"/p_test.txt")

# Scale datasets according to the test set distribution
sc = StandardScaler()

X_test = sc.fit_transform(X_test)
X_ref = sc.transform(X_ref)
X_train = sc.transform(X_train)

# Convert to torch tensors
X_ref = torch.from_numpy(X_ref).to(dtype=torch.float32)
X_train = torch.from_numpy(X_train).to(dtype=torch.float32)
X_test = torch.from_numpy(X_test).to(dtype=torch.float32)

Y_ref = torch.from_numpy(Y_ref).to(dtype=torch.float32).reshape(-1,1)
Y_train = torch.from_numpy(Y_train).to(dtype=torch.float32).reshape(-1,1)
Y_test = torch.from_numpy(Y_test).to(dtype=torch.float32).reshape(-1,1)

# Placeholder for falkon options
flk_opt = FalkonOptions()

# Number of distinct optimization runs from random initial bandwidths
nseeds = 100
# Number of epochs for the optimization
nepochs = 100

obs_out = np.zeros((nseeds, 4))

# Iterate over nseeds realizations
for se in range(nseeds):
    # Initialize the bandwidth vector
    sigma_exp = torch.FloatTensor(X_ref.shape[1], ).uniform_(0.0, 3.0)
    sigma_init = torch.pow(torch.full((len(sigma_exp),), 10), sigma_exp).requires_grad_()

    # Select a specific kernel form
    kernel = falkon.kernels.LaplacianKernel(sigma=sigma_init, opt=flk_opt)

    # Initialize regularization
    penalty_init = torch.tensor(1e-3, dtype=torch.float32)

    # Select all points of the design matrix
    # Here this can be reduced to form a Nystrom approximation
    centers_init = X_ref[np.random.choice(X_ref.shape[0], size=(X_ref.shape[0], ), replace=False)].clone()

    # The objective function form, here a simple L1 loss over a validation set
    model = SimpleVal(
        kernel=kernel, penalty_init=penalty_init, centers_init=centers_init,
        opt_penalty=True, opt_centers=False)

    # The optimizer
    opt_hp = torch.optim.Adam(model.parameters(), lr=1e1)

    re_err, tr_err, ts_err, lambda_evo = [], [], [], []

    # The actual optimization
    for epoch in range(nepochs):
        opt_hp.zero_grad()
        loss = model(X_ref, Y_ref, X_train, Y_train)

        loss.backward()
        opt_hp.step()

        lambda_evo.append(model.penalty.detach())
        re_err.append(mclass_loss(Y_ref, model.predict(X_ref)))
        tr_err.append(mclass_loss(Y_train, model.predict(X_train)))
        ts_err.append(mclass_loss(Y_test, model.predict(X_test)))

    print(f"Lambda: %.3e, Ref error: %.3e, Train error: %.3e, Test error: %.3e" % (model.penalty, re_err[-1], tr_err[-1], ts_err[-1]))

    # Final regularization and metrics
    obs_out[se, 0] = model.penalty.detach()
    obs_out[se, 1] = re_err[-1]
    obs_out[se, 2] = tr_err[-1]
    obs_out[se, 3] = ts_err[-1]

    # Saving the final bandwidth vector
    np.savetxt("results/"+subspace+"_bw_"+str(se)+".txt", model.kernel.sigma.detach())

    # Saving the epoch-resolved metrics
    np.savetxt("results/"+subspace+"_loss_ref_"+str(se)+".txt", np.asarray(re_err))
    np.savetxt("results/"+subspace+"_loss_train_"+str(se)+".txt", np.asarray(tr_err))
    np.savetxt("results/"+subspace+"_loss_test_"+str(se)+".txt", np.asarray(ts_err))

np.savetxt("results/"+subspace+"_lambda_loss.txt", obs_out)
