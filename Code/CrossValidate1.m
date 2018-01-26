% % Load the data
% load raw_tweets_train
% load train_cnn_feat
% load train_color
% load train_img_prob
% load train_raw_img
% load train_tweet_id_img
% load words_train
% XTextlabel = full(X);
% % load raw_tweets_train_unlabeled
% % load train_unlabeled_cnn_feat
% % load train_unlabeled_color
% % load train_unlabeled_img_prob
% % load train_unlabeled_raw_img
% % load train_unlabeled_tweet_id_img
% load words_train_unlabeled
% XTextunlabel = full(X);
% clear X

%% Get the feature
% % Text feature
% load pcaText
% reconstructAccuracy = 85;
% indPC = find(cumsum(varExplained)>reconstructAccuracy, 1);
% X = scoreTextAll(1:4500, :);
% Y = full(Y);
% XTextAll = [XTextlabel; XTextunlabel];

% % Image feature
% train_img_prob = round(train_img_prob*100);

% [coeff, scoreImage, ~, ~, varExplained] = pca(train_cnn_feat);
% reconstructAccuracy = 90;
% indPC = find(cumsum(varExplained)>reconstructAccuracy, 1);
% scoreImageChosen = scoreImage(:, 1:indPC);


%% Cross validation
Y = full(Y);
kFold=10;
cvInd=crossvalind('Kfold',size(Y,1),kFold);
decisionRule = 1; % 1: Majority vote
                  % 2: Regression
accuracyCross=zeros(1,10);
for i=1:10
    % Choose the train and test data
    foldI=i;
    trainLabel = Y(cvInd~=foldI);
    testLabel = Y(cvInd==foldI);
    
    %% Train models
%     % Naive Bayes
%     modelNB = fitcnb(train_color(cvInd~=foldI,:), trainLabel, 'Distribution','kernel');   
    
%     % SVM liblinear
%     modelSVM = train(trainLabel, sparse(train_cnn_feat(cvInd~=foldI,:)), '-c 1 -s 0');

%     % Logistic regression liblinear
%     modelLR = train(trainLabel, sparse(train_cnn_feat(cvInd~=foldI,:)),  '-c 0.1 -s 0');    

%     % Random fern
%     fernPrm = struct('S',10,'M',200,'thrr',[0 1],'bayes',1);
%     modelFern = fernsClfTrain(scoreImageChosen(cvInd~=foldI,:), trainLabel+1, fernPrm);

    % Adaboost matlab
    tic
    modelAda = fitensemble(train_cnn_feat(cvInd~=foldI,:), trainLabel, 'AdaBoostM1', 200, 'Tree');
    toc
    
%     % Random forest
%     nTrees = 200;
%     tic 
%     modelRF = TreeBagger(nTrees, train_cnn_feat(cvInd~=foldI,:), trainLabel,...
%         'MinLeafSize',1,'Method','classification');
%     toc
    
    %% Predict labels
%     yhatFinal = predict(modelNB, train_color(cvInd==foldI,:));   
%     yhatFinal = predict(ones(size(testLabel)), sparse( train_cnn_feat(cvInd==foldI,:)), modelSVM);  
%     [yhatFinal,probs] = fernsClfApply(scoreImageChosen(cvInd==foldI,:), modelFern);
%     yhatFinal = yhatFinal - 1;
%     logRatio = adaBoostApply(single(train_cnn_feat(cvInd==foldI,:)), modelAda);
%     yhatFinal = (logRatio>0);
%     testLabelEst = predict(modelRF, train_cnn_feat(cvInd==foldI,:));
%     yhatFinal = str2double(testLabelEst);
    yhatFinal = predict(modelAda, train_cnn_feat(cvInd==foldI,:));
    
    % Compute the accuracy
    accuracyCross(i) = mean(yhatFinal==testLabel);
    
    disp([num2str(i),'th turn:' num2str(accuracyCross(i))])
end

mean(accuracyCross)


% %% LDA cluster
% % Set the parameters
% K = 5; 
% BETA=0.01;
% ALPHA=50/K;
% N = 300; 
% OUTPUT = 0;
% 
% % Cluster the train set
% [phi, thetaTrain] = LDAssiBPtrain(sparse(XTextlabel(cvInd~=foldI,:)'), K, N, ALPHA, BETA, OUTPUT);    
% clusterTrain = thetaTrain';
% 
% % Train an SVM on the clustered train data
% modelSVM = train(trainLabel, sparse(clusterTrain), '-c 0.1 -s 0');
% 
% % Set the parameters
% N = 50; 
% 
% % Cluster the test set   
% thetaTest = LDAssiBPpredict(sparse(XTextlabel(cvInd==foldI,:)'), phi, N, ALPHA, BETA, OUTPUT);
% clusterTest = thetaTest';


N=5000; F=5000; sep=.01; RandStream.getGlobalStream.reset();
  [xTrn,hTrn,xTst,hTst]=demoGenData(N,N,2,F/10,sep,.5,0);
  xTrn=repmat(single(xTrn),[1 10]); xTst=repmat(single(xTst),[1 10]);
  pBoost=struct('nWeak',256,'verbose',16,'pTree',struct('maxDepth',2));
  model = adaBoostTrain( xTrn(hTrn==1,:), xTrn(hTrn==2,:), pBoost );
