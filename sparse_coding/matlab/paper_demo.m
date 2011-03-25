% maxNumCompThreads('automatic')
num_iters = 3; % 100

filename=sprintf('stats.txt');
fid=fopen(filename,'wt');
fprintf(fid,'n,k,batch_size,num_patches,matlab_time\n');

patch_dim = 32;
num_patches = 5000;
for beta =[1.0]
  for batch_size= [1000]
    for num_bases = [196]
        fs_time_start =cputime;
        demo_fast_sc(2,num_bases,num_iters,batch_size,patch_dim,num_patches,beta)
        fs_time_end = cputime;
        fprintf(fid, '%d, %d, %d, %d, %g, %g\n',num_bases,patch_dim*patch_dim,batch_size,num_patches,beta,fs_time_end-fs_time_start);
    end
  end
end
