clear all;
clc;
importfile('/Users/qiujingye/Downloads/ps4_2019_files/MNIST_data.mat')
C = 10;

disp('Start DAGSVM classification:')
% Create cells to store results
a_cell = cell(10);
b_cell = cell(10);
X_new_cell = cell(10);
Y_new_cell = cell(10);
classes = unique(train_samples_labels);
for i=1:10
    for j=i+1:10
        disp(['Training "',num2str(i-1),'" VS "',num2str(j-1),'"'])
        
        l1 = find(train_samples_labels == i-1);
        l2 = find(train_samples_labels == j-1);
        t1 = find(test_samples_labels == i-1);
        t2 = find(test_samples_labels == j-1);

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
        a_cell{i, j} = svm.a;
        b_cell{i, j} = b;
        X_new_cell{i, j} = svm.Xsv;
        Y_new_cell{i, j} = svm.Ysv;
    end
end

disp('Testing...');
pred = zeros(size(test_samples, 1), 1);
for i = 1:size(test_samples, 1)
    remain = classes; % From 0 to 9
    while length(remain) > 1
        a = a_cell{remain(1) + 1, remain(end) + 1};
        b = b_cell{remain(1) + 1, remain(end) + 1};
        X_new = X_new_cell{remain(1) + 1, remain(end) + 1};
        Y_new = Y_new_cell{remain(1) + 1, remain(end) + 1};
        out = (a' .* Y_new) * kernel(X_new, test_samples(i, :)') + b;
        if out > 0
            remain = remain(1:end-1); % Remove the tail No. of the remain
        else
            remain = remain(2:end); % Remove the head No. of the remain
        end
    end
    % What is left wins
    pred(i) = remain;
end
accuracy = length(find(pred==test_samples_labels))/length(test_samples_labels);
disp(['For DAGSVM, the accuracy is ',num2str(accuracy,10)])
disp(' ')