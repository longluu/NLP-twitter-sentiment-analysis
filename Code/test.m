
% load raw_tweets_train
% load train_cnn_feat
% load train_color
% load train_img_prob
% load train_raw_img
% load train_tweet_id_img
% load words_train
% XTextlabel = full(X);
% 
% Y_hat= predict_labels(XTextlabel, train_cnn_feat, train_img_prob, train_color, train_img, raw_tweets_train_unlabeled);
% sum(Y_hat == full(Y))/ length(Y)

% Extract the posterior of most frequent happy and sad words
posteriorNew = posteriorNewOriginal;
pSad = posteriorNew(:, 1);
pHappy = posteriorNew(:, 2);
[~, indexSad] = sort(pSad, 1, 'descend');
[~, indexHappy] = sort(pHappy, 1, 'descend');
[~, indexNeutral] = sort(abs(pSad-0.5), 1, 'ascend');

% Display the words
nWords = 25;
figure
hold on
fontSizeAve = 20;
fontSizeMax = fontSizeAve + nWords/2;
for ii = 1 : nWords
    plot(nWords-ii+3, 1);
    text(nWords-ii+3, 1, wordVocab{indexSad(ii)}, 'Rotation', -90, 'FontSize', fontSizeMax-ii+1, 'Color', 'blue');
    plot(1, nWords-ii+3);
    text(1, nWords-ii+3, wordVocab{indexHappy(ii)}, 'FontSize', fontSizeMax-ii+1, 'Color', 'red'); 
    plot(nWords-ii+3, nWords-ii+3);
    text(nWords-ii+3, nWords-ii+3, wordVocab{indexNeutral(ii)}, 'FontSize', fontSizeMax-ii+1, 'Color', 'black');
end
xlabel('Sadness index')
ylabel('Happiness index')
xlim([0 nWords+6])
ylim([-4 nWords+3])
set(gca, 'FontSize', 23)
