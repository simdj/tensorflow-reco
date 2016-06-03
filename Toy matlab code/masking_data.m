function ret = masking_data(x,p)
ret=x;
[row_len,col_len]=size(ret);
for i=1:row_len
    for j=1:col_len
        if rand(1,1)>p
            ret(i,j)=0;
        end
    end
end