function yHat = logisticRegression(Xtrain, Ytrain, Xtest)
[B] = mnrfit(Xtrain, Ytrain+1);
pY = mnrval(B, Xtest);
[~, yHat] = max(pY, [] ,2);
yHat = yHat - 1;

% lrModel = fitclinear(Xtrain, Ytrain, 'Learner', 'logistic','Regularization','ridge', 'Lambda', 0.00001);
% yHat = predict(lrModel, Xtest);
end