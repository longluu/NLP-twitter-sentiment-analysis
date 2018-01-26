function yHat = sparseLR(Xtrain, Ytrain, Xtest)
w = smlr(Xtrain, Ytrain);
pY = mnrval(w, Xtest);
[~, yHat] = max(pY, [] ,2);
