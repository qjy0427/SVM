clear all;
clc;
importfile('/Users/qiujingye/Downloads/ps4_2019_files/MNIST_data.mat')
C = 10;

disp('Start one-vs-one classification:')
% Create cells to store results
a_cell = cell(10);
b_cell = cell(10);
X_new_cell = cell(10);
Y_new_cell = cell(10);
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
        % Store results into cells
        a_cell{i, j} = svm.a;
        b_cell{i, j} = b;
        X_new_cell{i, j} = svm.Xsv;
        Y_new_cell{i, j} = svm.Ysv;
        
    end
end

disp('Testing...');
pred = zeros(size(test_samples, 1), 1);
for i = 1:size(test_samples, 1)
    vote = zeros(10, 1);
    for j = 1:10
        for k = j+1:10
            a = a_cell{j,k};
            b = b_cell{j,k};
            X_new = X_new_cell{j,k};
            Y_new = Y_new_cell{j,k};
            out = (a' .* Y_new) * kernel(X_new, test_samples(i, :)') + b;
            % The one that has positive outcome gets a vote
            if out > 0
                vote(j,1) = vote(j,1) + 1;
            else
                vote(k,1) = vote(k,1) + 1;
            end
        end
    end
    % The one gets the most votes is the winner
    [x,I] = max(vote);
    pred(i) = I-1;
end
accuracy = length(find(pred==test_samples_labels))/length(test_samples_labels);
disp(['For one-vs-one, the accuracy is ',num2str(accuracy)])
disp(' ')