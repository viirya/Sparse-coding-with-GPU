function demo_fast_sc(opt_choice,num_bases,num_iters,batch_size,patch_dim,num_patches,beta)
% opt_choice = 1: use epslion-L1 penalty
% opt_choice = 2: use L1 penalty

if ~exist('opt_choice', 'var')
    opt_choice = 1; 
end

opt_choice = 2; % 

% natural image data
load ../../images/IMAGES.mat
% X = getdata_imagearray(IMAGES, 14, 10000);
X = getdata_imagearray(IMAGES, patch_dim, num_patches);

% sparse coding parameters
% beta = 0.4;
if opt_choice==1
    sparsity_func= 'epsL1';
    epsilon = 0.01;
elseif opt_choice==2
    sparsity_func= 'L1';
    epsilon = [];
end

Binit = [];
fname_save = sprintf('results/sc_%s_b%d_beta%g_%s', sparsity_func, num_bases, beta, datestr(now, 30));	

% run fast sparse coding
[B S stat] = sparse_coding(X, num_bases, beta, sparsity_func, epsilon, num_iters, batch_size, fname_save, Binit);
