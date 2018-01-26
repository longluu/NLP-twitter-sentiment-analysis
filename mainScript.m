%%%%%%%%%%%%%%%%%%%%%% Main script %%%%%%%%%%%%%%%%%%%%%
clear

% Load the data
load raw_tweets_train
load train_cnn_feat
load train_color
load train_img_prob
load train_raw_img
load train_tweet_id_img
load words_train
XTextlabel = full(X);
% load raw_tweets_train_unlabeled
% load train_unlabeled_cnn_feat
% load train_unlabeled_color
% load train_unlabeled_img_prob
% load train_unlabeled_raw_img
% load train_unlabeled_tweet_id_img
load words_train_unlabeled
Xunlabeled = full(X);
clear X


%% Train on the text feature
% % Do PCA on the text feature
% tic
% [coeff, scoreTextAll, ~, ~, varExplained] = pca([Xlabel; Xunlabeled]);
% toc
load pcaText

% Pick the number of PCs
reconstructAccuracy = 85;
figure
plot(cumsum(varExplained))
xlabel('Number of PCs')
ylabel('Variance explained')
title([num2str(find(cumsum(varExplained)>reconstructAccuracy, 1)) ' PCs to get ' num2str(reconstructAccuracy) '% accurate'])

% Train the SVM on the PCA scores
indPC = find(cumsum(varExplained)>reconstructAccuracy, 1);
labelTrain = full(Y);
tic
SVMModel = fitcsvm(scoreTextAll(1:4500, 1:indPC), labelTrain, 'KernelFunction', 'linear', ...
                    'KernelScale', 'auto', 'Standardize', false, 'CrossVal', 'on'); 
toc

% The cross-validation accuracy 
accuracyCross = 1-kfoldLoss(SVMModel, 'lossfun', 'classiferror')

%% Naive Bayes
train_img_prob = round(train_img_prob*100);
modelNB = fitcnb(train_img_prob, Y, 'Distribution', 'mn'); 

%% Logistic regression
modelLR = train(Y, sparse(XTextlabel),  '-c 0.4 -s 0'); 

%% SVM matlab
modelSVM = fitcsvm(XTextlabel, Y, 'KernelFunction', 'linear', ...
                'KernelScale', 'auto', 'Standardize', false); 

%% SVM liblinear
modelSVM = train(Y, sparse(XTextlabel), '-c 0.1 -s 1');

%% Fern
fernPrm = struct('S',10,'M',200,'thrr',[0 1],'bayes',1);
modelFern = fernsClfTrain(train_cnn_feat(cvInd~=foldI,:), trainLabel+1, fernPrm);
[yhatFinal,probs] = fernsClfApply(train_cnn_feat(cvInd==foldI,:), modelFern);
yhatFinal = yhatFinal - 1;

%% AdataBoost
Xtrain = single(train_cnn_feat);
pBoost=struct('nWeak',200,'pTree',struct('maxDepth',1), 'discrete', 0);
modelAda = adaBoostTrain( Xtrain(Y==0,:), Xtrain(Y==1,:), pBoost );

%% Ensemble the methods
train_img_prob = round(train_img_prob*100);
% Get probability/score/decision for the train set
[~, postProbNB ,~] = predict(modelNB, train_img_prob);
[~, ~, scoresSVM] = predict(ones(size(XTextlabel, 1), 1), sparse( XTextlabel), modelSVM);
[~, ~, probLR] = predict(ones(size(XTextlabel, 1), 1), sparse( XTextlabel), modelLR, '-q  -b 1');
logRatio = double(adaBoostApply(single(train_cnn_feat), modelAda));

% 
% Fit a LR model
modelDecision = train(Y, sparse([postProbNB(:,1) scoresSVM(:,1) probLR(:,1) logRatio]),  '-c 2 -s 0');

                
%% Train on the image and/or image category feature
reconstructAccuracy = 40;

% Combine the image and image category feature
zScoreCNN = zscore([train_cnn_feat; train_unlabeled_cnn_feat]);
zScoreImgProb = zscore([log(train_img_prob); log(train_unlabeled_img_prob)]);
Xtrain = [zScoreCNN];
       
% Do PCA on the color feature
[coeff, scoreImage, ~, ~, varExplained] = pca(Xtrain);

figure
plot(cumsum(varExplained))
xlabel('Number of PCs')
ylabel('Variance explained')
title([num2str(find(cumsum(varExplained)>reconstructAccuracy, 1)) ' PCs to get ' num2str(reconstructAccuracy) '% accurate'])

% Train the SVM on the PCA scores
indPC = find(cumsum(varExplained)>reconstructAccuracy, 1);
labelTrain = full(Y);
SVMModel = fitcsvm(scoreImage(1:4500, 1:indPC), labelTrain, 'KernelFunction', 'linear', ...
                    'KernelScale', 'auto', 'Standardize',true, 'CrossVal','on');

% The cross-validation accuracy 
accuracyCross = 1-kfoldLoss(SVMModel, 'lossfun', 'classiferror')

%% Check the data 
% % Get the index of joy tweets
% indJoy = find(Y==1);
% 
% % Retrieve the joy verbal tweets
% indJoyRaw = find(ismember(raw_tweets_train{1,1}, tweet_ids(indJoy)));
% verbTweetJoy = raw_tweets_train{1,2}(indJoyRaw);
% 
% % Align the verbal and image tweets
% indJoyVerbTweet = find(ismember(tweet_ids, raw_tweets_train{1,1}(indJoyRaw)));
% 
% % Retrive the joy image tweets
% imageTweetJoy = train_img(indJoy, :);
% 
% % Display some joy image and verbal tweets
% n = 40;
% nRow = round(sqrt(n));
% nColumn = ceil(n/nRow);
% figure
% for ii = 1 : n
%     imageDisplay =  reshape_img(imageTweetJoy(ii, :));
%     subplot(nRow, nColumn, ii)
%     imagesc(imageDisplay)
% %     title(verbTweetJoy{ii})
%     axis off
% end

