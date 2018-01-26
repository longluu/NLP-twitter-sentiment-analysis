% % Load the data
% load raw_tweets_train
% load train_cnn_feat
% load train_color
% load train_img_prob
% load train_raw_img
% load train_tweet_id_img
% load words_train
% XTextlabel = full(X);
% addpath('./liblinear')

%% Cross validation
% Y = full(Y);
% kFold=10;
% cvInd=crossvalind('Kfold',size(Y,1),kFold);
accuracyCross=zeros(1,10);
regularizationWeight = [0.01 0.03 0.05 0.1 0.3 0.5 0.7 0.9];
accuracyAll = NaN(1, length(regularizationWeight));
for kk = 1:length(regularizationWeight)
    for i=1:10
        % Choose the train and test data
        foldI=i;
        trainLabel = Y(cvInd~=foldI);
        testLabel = Y(cvInd==foldI);

        %% Train models
        % SVM liblinear
        modelSVM = train(trainLabel, sparse(XTextlabel(cvInd~=foldI,:)), ['-c ' num2str(regularizationWeight(kk)) ' -s 5']);

        %% Predict labels
        yhatFinal = predict(ones(size(testLabel)), sparse( XTextlabel(cvInd==foldI,:)), modelSVM);  

        % Compute the accuracy
        accuracyCross(i) = mean(yhatFinal==testLabel);

        disp([num2str(i),'th turn:' num2str(accuracyCross(i))])
    end
    accuracyAll(kk) = mean(accuracyCross);  
end


%% Plot the accuracy
accuracyL2RegL2Loss = accuracyAll;
accuracyL2RegL1Loss = accuracyAll;
accuracyL1RegL2Loss = accuracyAll;

figure
hold on
plot(regularizationWeight, 100*accuracyL2RegL2Loss, 'ro-')
plot(regularizationWeight, 100*accuracyL2RegL1Loss, 'gx-')
plot(regularizationWeight, 100*accuracyL1RegL2Loss, 'b+-')

xlabel('Regularization weight')
ylabel('Accuracy (%)')
legend('L2 Regularization, L2 loss', 'L2 Regularization, L1 loss', 'L1 Regularization, L2 loss', 'Location', 'SouthEast')
set(gca, 'FontSize', 20)

%% The support vectors
% Import the word vocabulary
[num, wordVocab] = xlsread('topwords.csv');
modelSVM = train(Y, sparse(XTextlabel), ['-c 0.05 -s 1']);
weight = modelSVM.w;
indWeigthPos = weight>0;
indWeigthNeg = weight<0;
weigthPositive = weigth(indWeigthPos);
weigthNegative = weigth(indWeigthNeg);
[~, indexPos] = sort(weigthPositive, 1, 'descend');
[~, indexNeg] = sort(weigthNegative, 1, 'descend');

