clear all;
clc;
importfile('/Users/qiujingye/Downloads/ps4_2019_files/MNIST_data.mat')
C = 10;

disp('Start one-vs-the-rest classification:')
% Create cells to store results
a_cell = cell(10);
b_cell = cell(10);
X_new_cell = cell(10);
Y_new_cell = cell(10);
for k=0:9
    disp(['Training "',num2str(k),'" VS the rest'])
    l1 = find(train_samples_labels == k);
    l2 = find(train_samples_labels(:,1) ~= k);
    t1 = find(test_samples_labels == k);
    t2 = find(test_samples_labels(:,1) ~= k);

    x1 = train_samples(l1,:)';
    y1 = ones(1,length(l1));
    x2 = train_samples(l2,:)';
    y2 = -ones(1,length(l2));

    X = [x1,x2];
    Y = [y1,y2];
    svm = svmTrain(X,Y,C);
    temp = (svm.a'.*svm.Ysv)*kernel(svm.Xsv,svm.Xsv);
    total_b = svm.Ysv-temp;
    b = mean(total_b);
    % Store results to cells
    a_cell{k+1, 1} = svm.a;
    b_cell{k+1, 1} = b;
    X_new_cell{k+1, 1} = svm.Xsv;
    Y_new_cell{k+1, 1} = svm.Ysv;
end

disp('Testing...');
pred = zeros(size(test_samples, 1), 1);
for i = 1:size(test_samples, 1)
    score = zeros(10, 1);
    for j = 0:9
            a = a_cell{j+1, 1};
            b = b_cell{j+1, 1};
            X_new = X_new_cell{j+1, 1};
            Y_new = Y_new_cell{j+1, 1};
            out = (a' .* Y_new) * kernel(X_new, test_samples(i, :)') + b;
            score(j+1) = out;
    end
    % The one that gets the highest score wins
    [x, I] = max(score);
    pred(i) = I-1;
end

accuracy = length(find(pred==test_samples_labels))/length(test_samples_labels);
disp(['For one-vs-the-rest, the accuracy is ',num2str(accuracy)])
disp(' ')