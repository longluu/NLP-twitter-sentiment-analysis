% Load the data
load raw_tweets_train
load train_cnn_feat
load train_color
load train_img_prob
load train_raw_img
load train_tweet_id_img
load words_train
XTextlabel = full(X);

%% Cross validation
Y = full(Y);
kFold=10;
cvInd=crossvalind('Kfold',size(Y,1),kFold);
accuracyCross=zeros(1,10);
train_img_prob = round(100*train_img_prob);
for i=1:10
    % Choose the train and test data
    foldI=i;
    trainLabel = Y(cvInd~=foldI);
    testLabel = Y(cvInd==foldI);
    
    %% Train models
    % Naive Bayes
    modelNB = fitcnb(XTextlabel(cvInd~=foldI,:), trainLabel, 'Distribution', 'mn');       
    
    %% Predict labels
    yhatFinal = predict(modelNB, XTextlabel(cvInd==foldI,:));   
    
    % Compute the accuracy
    accuracyCross(i) = mean(yhatFinal==testLabel);
    
    disp([num2str(i),'th turn:' num2str(accuracyCross(i))])
end

mean(accuracyCross)

%% Visualize naive Bayes
% Fit the whold dataset
modelNB = fitcnb(XTextlabel, Y, 'Distribution', 'mn');

% Predict the posterior for each word
Xnew = diag(ones(1, 10000));
[Ynew, posteriorNew] = predict(modelNB, Xnew);
posteriorNewOriginal = posteriorNew;

% Import the word vocabulary
[num, wordVocab] = xlsread('topwords.csv');

% Extract the posterior of most frequent happy and sad words
posteriorNew = 300.^(posteriorNewOriginal);
pSad = posteriorNew(:, 1);
pHappy = posteriorNew(:, 2);
[~, indexSad] = sort(pSad, 1, 'descend');
[~, indexHappy] = sort(pHappy, 1, 'descend');

% Display the words
nWords = 20;
figure
hold on
for ii = 1 : nWords
    plot(posteriorNew(indexSad(ii), 1),posteriorNew(indexSad(ii), 2));
    text(posteriorNew(indexSad(ii), 1),posteriorNew(indexSad(ii), 2), wordVocab{indexSad(ii)});
    plot(posteriorNew(indexHappy(ii), 1),posteriorNew(indexHappy(ii), 2));
    text(posteriorNew(indexHappy(ii), 1),posteriorNew(indexHappy(ii), 2), wordVocab{indexHappy(ii)});    
end
xlabel('Posterior probability P({sad}|word)')
ylabel('Posterior probability P({happy}|word)')
