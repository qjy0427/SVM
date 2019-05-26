function importfile(fileToRead1)

newData1 = load('-mat', fileToRead1);

vars = fieldnames(newData1);
for i = 1:length(vars)
    assignin('base', vars{i}, newData1.(vars{i}));
end
