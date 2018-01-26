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
XTextunlabel = full(X);
clear X

%% Get the feature
% Text feature
% load pcaText
% reconstructAccuracy = 85;
% indPC = find(cumsum(varExplained)>reconstructAccuracy, 1);
% X = scoreTextAll(1:4500, :);
Y = full(Y);
% 
% % Image feature
train_img_prob = round(train_img_prob*100);

%% Cross validation
kFold=10;
cvInd=crossvalind('Kfold',size(Y,1),kFold);
decisionRule = 2; % 1: Majority vote
                  % 2: Regression
accuracyCross=zeros(1,10);
for i=1:10
    % Choose the train and test data
    foldI=i;
    trainLabel = Y(cvInd~=foldI);
    testLabel = Y(cvInd==foldI);
    
    % Naive Bayes
    modelNB = fitcnb(train_img_prob(cvInd~=foldI,:), trainLabel, 'Distribution','mn');   
    
    % SVM liblinear
    modelSVM = train(trainLabel, sparse(XTextlabel(cvInd~=foldI,:)), '-c 0.1 -s 1');
      
    % Logistic regression liblinear
    modelLR = train(trainLabel, sparse(XTextlabel(cvInd~=foldI,:)),  '-c 0.4 -s 0');    

%     % Random fern
%     fernPrm = struct('S',10,'M',200,'thrr',[0 1],'bayes',1);
%     modelFern = fernsClfTrain(train_color(cvInd~=foldI,:), trainLabel+1, fernPrm);

%     % Adaboost
%     Xtrain = single(train_cnn_feat(cvInd~=foldI,:));
%     pBoost=struct('nWeak',200,'pTree',struct('maxDepth',1), 'discrete', 0);
%     modelAda = adaBoostTrain( Xtrain(trainLabel==0,:), Xtrain(trainLabel==1,:), pBoost );

    % Decision rule
    if decisionRule == 1
        % Majority vote
        yHatNB = predict(modelNB, train_img_prob(cvInd==foldI,:));
        yHatSVM = predict(ones(size(testLabel)), sparse( XTextlabel(cvInd==foldI,:)), modelSVM);
        yHatLR = predict(ones(size(testLabel)), sparse( XTextlabel(cvInd==foldI,:)), modelLR, '-q');
        yhatFinal = (yHatSVM)>=0.5; %  (yHatNB + yHatSVM + yHatLR)/3
    elseif decisionRule == 2
        % Get probability/score/decision for the train set
        [~, postProbNB ,~] = predict(modelNB, train_img_prob(cvInd ~= foldI,:));
        [~, ~, scoresSVM] = predict(ones(size(trainLabel)), sparse( XTextlabel(cvInd ~= foldI,:)), modelSVM);
        [~, ~, probLR] = predict(ones(size(trainLabel)), sparse( XTextlabel(cvInd ~= foldI,:)), modelLR, '-q  -b 1');
%         [yFern, ~] = fernsClfApply(train_cnn_feat(cvInd~=foldI,:), modelFern);
%         logRatio = double(adaBoostApply(single(train_cnn_feat(cvInd~=foldI,:)), modelAda));
        [~, scoresFR] = predict(modelRF, train_cnn_feat(cvInd~=foldI,:));
        
        % Fit a decision model  
        trainModelOut = [postProbNB(:,1) scoresSVM(:,1) probLR(:,1)  scoresFR(:,1)];
        modelDecision = train(trainLabel, sparse(trainModelOut),  '-c 5 -s 0');
                
        % Get probability/score/decision for the test set
        [~, postProbNB_test ,~] = predict(modelNB, train_img_prob(cvInd==foldI,:));
        [~, ~, scoresSVM_test] = predict(ones(size(testLabel)), sparse( XTextlabel(cvInd == foldI,:)), modelSVM);
        [~, ~, probLR_test] = predict(ones(size(testLabel)), sparse( XTextlabel(cvInd==foldI,:)), modelLR, '-q  -b 1');
%         [yFern_test, ~] = fernsClfApply(train_cnn_feat(cvInd==foldI,:), modelFern);
%         logRatio_test = double(adaBoostApply(single(train_cnn_feat(cvInd==foldI,:)), modelAda));
        [~, scoresFR_test] = predict(modelRF, train_cnn_feat(cvInd==foldI,:));
        
        % Predict the test label
        testModelOut = [postProbNB_test(:,1) scoresSVM_test(:,1) probLR_test(:,1) scoresFR_test(:,1)];
%         yhatFinal = predict(ones(size(testLabel)), ...
%                sparse([postProbNB_test(:,1) scoresSVM_test(:,1) probLR_test(:,1) probFern_test(:,1)]), modelDecision, '-q');
        yhatFinal = predict(ones(size(testLabel)), ...
               sparse(testModelOut), modelDecision, '-q');
   
    end

    accuracyCross(i)=mean(yhatFinal==testLabel);
    
    disp([num2str(i),'th turn:' num2str(accuracyCross(i))])
end

mean(accuracyCross)