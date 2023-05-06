clear all; close all;

warning off
%% Load the sharps boundary conditions
allsharps = 7050:8369;
sharps = [];
for j = 1:numel(allsharps)
    sharp = allsharps(j);
    fileList = dir(sprintf( ...
        "D:\\nats ML stuff\\SHARP_data_v2\\hmi.sharp_cea_720s.%i.*.fits" ...
        , sharp));
    matList = dir(sprintf( ...
        "D:\\nats ML stuff\\MHS_solutions_v3\\sharp%i.mat" ...
        , sharp));
    if numel(fileList) > 0 && numel(matList) == 0
        fprintf("%i\n",j);
        splitName = split(fileList(1).name,"."); datetime = splitName{4};
        filename = sprintf(...
            "D:\\nats ML stuff\\SHARP_data_v2\\hmi.sharp_cea_720s.%i.%s", ...
            sharp,datetime);
        trial = fitsread(filename+".Br.fits","image");
        if max(abs(trial(:))) < 1e4
            Brs{j} = fitsread(filename + ".Br.fits","image");
            Bps{j} = fitsread(filename + ".Bp.fits","image");
            Bts{j} = fitsread(filename + ".Bt.fits","image");
            
            sharps = [sharps,j];
        end
    end

end

%% Choose a sharps to evaluate
sharpsn = numel(sharps);
availableGPUs = gpuDeviceCount("available");

for newj = 836:-1:1
    tic;

    j = sharps(newj);
    parprint(sprintf("j=%i/%i starting...\n",newj,sharpsn));

    Br = Brs{j}(end:-1:1,:)/1e3;
    Bt = Bts{j}(end:-1:1,:)/1e3;
    Bp = Bps{j}(end:-1:1,:)/1e3;
    % Build the nodeset
    numpoints = 30;
    
    supersmooth = GaussianFilter(abs(Br),63,15);
    supersmooth = supersmooth / max(supersmooth(:));
    imageX = size(Br,1)/min(size(Br)); imageY = size(Br,2)/min(size(Br));
    height = 2*max(imageX,imageY);


    [X,Y] = ndgrid(linspace(0,imageX,size(Br,1)),linspace(0,imageY,size(Br,2)));
    electrofn = griddedInterpolant(X,Y,5./(5*supersmooth.^2+1));
    
    [nodesx,nodesy,nodesz] = meshgrid(linspace(0,1,numpoints));
    nodes = [nodesx(:),nodesy(:),nodesz(:)];
    nodes(:,1) = nodes(:,1)*imageX;
    nodes(:,2) = nodes(:,2)*imageY;
    nodes(:,3) = nodes(:,3)*height;
    
    constantfn = @(x,y,z)ones(size(x));
    impulse = 2e-4;
    nodes = unique(electrostatic_repulsion(nodes,20, ...
        constantfn,impulse,[0 imageX 0 imageY 0 height],30),'rows');

    electrofn3d = @(x,y,z)(electrofn(x,y).*exp(4*z/height));
    impulse = 2e-5;
    nodes = unique(electrostatic_repulsion(nodes,100, ...
        electrofn3d,impulse,[0 imageX 0 imageY 0 height],30),'rows');
    
    eps = 1e-2;
    index_x0 = find(nodes(:,1)<eps*imageX);
    index_x1 = find(nodes(:,1)>imageX-eps*imageX);
    index_y0 = find(nodes(:,2)<eps*imageY);
    index_y1 = find(nodes(:,2)>imageY-eps*imageY);
    index_z0 = find(nodes(:,3)<eps*height);
    index_z1 = find(nodes(:,3)>height-eps*height);
    
    nodes = [nodes;nodes(index_z0,:)-[0,0,0.5*height/numpoints]];
    n = size(nodes,1);
    
    index_gh = ((n-numel(index_z0)+1):n)';
    
    index_bd = unique([index_z0;index_z1;index_gh;index_x0;index_x1;index_y0;index_y1]);
    index_in = setdiff(1:n,index_bd);
    
    % Build RBF matrices
    rbfk = 100;
    rbfk_bd = 180;
    rbfd = 5;
    rbfm = 4;
    
    idx_in = knnsearch(nodes,nodes(index_in,:),'k',rbfk);
    idx_bd = knnsearch(nodes, nodes(index_bd,:),'k',rbfk_bd);
    w_in = zeros(numel(index_in),rbfk, 10);
    w_bd = zeros(numel(index_bd),rbfk_bd, 10);
    for k = 1:numel(index_in)
        xx = nodes(idx_in(k,:),1);
        yy = nodes(idx_in(k,:),2);
        zz = nodes(idx_in(k,:),3);
        w_in( k,:,: ) = RBF_FD_PHS_pol_weights_3D (xx,yy,zz,rbfd,rbfm);
        if isnan(squeeze(w_in(k,:,:)))
            keyboard();
        end
    end
    for k = 1:numel(index_bd)
        xx = nodes(idx_bd(k,:),1);
        yy = nodes(idx_bd(k,:),2);
        zz = nodes(idx_bd(k,:),3);
        w_bd( k,:,: ) = RBF_FD_PHS_pol_weights_3D(xx,yy,zz,rbfd,rbfm);
        if isnan(squeeze(w_bd(k,:,:)))
            keyboard();
        end
    end
    w_in = w_in(:,:,2:end);
    w_bd = w_bd(:,:,2:end);
    
    Dx = sparse(repmat(idx_in(:,1),1,rbfk   ),idx_in,w_in(:,:,1),n,n) + ...
         sparse(repmat(idx_bd(:,1),1,rbfk_bd),idx_bd,w_bd(:,:,1),n,n);
    Dy = sparse(repmat(idx_in(:,1),1,rbfk   ),idx_in,w_in(:,:,2),n,n) + ...
         sparse(repmat(idx_bd(:,1),1,rbfk_bd),idx_bd,w_bd(:,:,2),n,n);
    Dz = sparse(repmat(idx_in(:,1),1,rbfk   ),idx_in,w_in(:,:,3),n,n) + ...
         sparse(repmat(idx_bd(:,1),1,rbfk_bd),idx_bd,w_bd(:,:,3),n,n);
    Dxx = sparse(repmat(idx_in(:,1),1,rbfk   ),idx_in,w_in(:,:,4),n,n) + ...
          sparse(repmat(idx_bd(:,1),1,rbfk_bd),idx_bd,w_bd(:,:,4),n,n);
    Dyy = sparse(repmat(idx_in(:,1),1,rbfk   ),idx_in,w_in(:,:,6),n,n) + ...
          sparse(repmat(idx_bd(:,1),1,rbfk_bd),idx_bd,w_bd(:,:,6),n,n);
    Dzz = sparse(repmat(idx_in(:,1),1,rbfk   ),idx_in,w_in(:,:,9),n,n) + ...
          sparse(repmat(idx_bd(:,1),1,rbfk_bd),idx_bd,w_bd(:,:,9),n,n);

    % Set parameters for run
    
    m = 8;
    sampleN = 6;
    interpN = 64;
    pressn = 100;
    Bns = zeros(3*n,sampleN);
    rns = zeros(sampleN);
    forcevec = 0*Bns;
    gamma = 1e-2;

    Bx = griddedInterpolant(X,Y,-Bp);
    By = griddedInterpolant(X,Y,Bt);
    Bz = griddedInterpolant(X,Y,Br);
    Bx0 = Bx(nodes(index_z0,1),nodes(index_z0,2));
    By0 = By(nodes(index_z0,1),nodes(index_z0,2));
    Bz0 = Bz(nodes(index_z0,1),nodes(index_z0,2));
    
    samplepresnode = net(haltonset(3),pressn);
    [~,pDx,pDy,pDz] = RBF_find_global_weights( ...
                samplepresnode(:,1),samplepresnode(:,2),samplepresnode(:,3),...
                nodes(index_z0,1),nodes(index_z0,2),nodes(index_z0,3), ...
                3, 0 ...
               );
    [Q,R] = qr((pDx.*Bx0+pDy.*By0+pDz.*Bz0)');
    [r,rd] = sort(abs(diag(R)));
    sd = randperm(max(find(r>=1e-3,1),sampleN),sampleN);
    samplepressure = 4*Q(:,rd(sd));

    [Bpx,Bpy,Bpz] = potfield(nodes,index_z0,Bz0,64);
    [Bff,rff] = num_mhs(zeros(n,1),zeros(n,3),Bx0,By0,Bz0, n, ...
            nodes, Dx, Dy, Dz, Dxx, Dyy, Dzz, 1,...
            index_x0,index_x1,index_y0,index_y1,index_z0,index_z1,index_gh, ...
            gamma,[Bpx;Bpy;Bpz],[Bpx;Bpy;Bpz]);
    parprint(sprintf('Generated force-free version r=%s.\n',sprintf(" %e ",rff)));
    
    for cpuj = 1:sampleN
    
        presmat = RBF_global_weights_3D( ...
                    samplepresnode(:,1),samplepresnode(:,2),samplepresnode(:,3),...
                    samplepressure(:,cpuj),nodes(:,1),nodes(:,2),nodes(:,3), ...
                    3, 0 ...
                   );

        pres = presmat(:,1);
        presx = presmat(:,2);
        presy = presmat(:,3);
        presz = presmat(:,4);
        
        dens = zeros(n,1);
    
        [Bns(:,cpuj),rs] = num_mhs(dens,pres,Bx0,By0,Bz0, n, ...
            nodes, Dx, Dy, Dz, Dxx, Dyy, Dzz, 1,...
            index_x0,index_x1,index_y0,index_y1,index_z0,index_z1,index_gh, ...
            gamma,Bff,[Bpx;Bpy;Bpz]);
        rns(cpuj) = rs(end);

        forcevec(:,cpuj) = [presx;presy;presz];

        parprint(sprintf(" -%i - %i/%i done, r=%s\n",newj,cpuj,sampleN,sprintf(" %e ",rs)));
    end
    parprint(sprintf("j=%i/%i done in %f.\n",newj,sharpsn,toc));
    parsave(sprintf("D:\\nats ML stuff\\MHS_solutions_v3\\sharp%i.mat", ...
        allsharps(j)),Bns,nodes,index_z0,n,rns,forcevec,Bff);
end
%%
for plotind = 0:6
    if plotind == 0, Bn = Bff; else, Bn = Bns(:,plotind); end
    % Plotting
    numX = scatteredInterpolant(nodes, Bn(1:n),'nearest');
    numY = scatteredInterpolant(nodes, Bn(n+1:2*n),'nearest');
    numZ = scatteredInterpolant(nodes, Bn(2*n+1:3*n),'nearest');
    
    step = 0.01;
    [xq,yq,zq] = meshgrid(0:step:max(nodes(:,1)),0:step:0:step:max(nodes(:,2)),0:step:max(nodes(:,3)));
    Bxnq = numX(xq,yq,zq);
    Bynq = numY(xq,yq,zq);
    Bznq = numZ(xq,yq,zq);
    
    % Streamlines
    step = 8;
    startx = squeeze(xq(1:step:end,1:step:end,1));
    starty = squeeze(yq(1:step:end,1:step:end,1));
    startz = squeeze(zq(1:step:end,1:step:end,1));
    
    fig = figure(plotind+1);
    ax = axes('Parent',fig);
    h1 = streamline(xq,yq,zq,Bxnq,Bynq,Bznq,startx(:),starty(:),startz(:));
    hold on;
    h2 = streamline(xq,yq,zq,-Bxnq,-Bynq,-Bznq,startx(:),starty(:),startz(:));
    h3 = slice(xq,yq,zq,Bznq,[],[],0);
    colormap('jet');
    set(h1,'Color','k'); set(h2, 'Color','k'); set(h3,'edgecolor','flat');
    view(3); %axis([0 1 0 1 0 1]);
    set(gca,'Fontsize',16);
    view(ax,[-1.1,23.8]);
    xlabel('x'); ylabel('y'); zlabel('z');
    hold off;
end


%% functions
function parsave(fname, Bns,nodes,index_z0,n,rns,forcevec,Bff)
 save(fname,"Bns","nodes","index_z0","n","rns","forcevec","Bff");
end

function parprint(string)
    fprintf(string);
end