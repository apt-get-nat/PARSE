clear all; close all;

warning off
%% Load the sharps boundary conditions
allsharps = 8369:9678;%7050:8369;
sharps = [];
for j = 1:numel(allsharps)
    sharp = allsharps(j);
    fileList = dir(sprintf( ...
        "D:\\SHARP_data_v3\\hmi.sharp_cea_720s.%i.*.fits" ...
        , sharp));
    matList = dir(sprintf( ...
        "D:\\MHS_solutions_v4\\sharp%i.mat" ...
        , sharp));
    if numel(fileList) > 0 && numel(matList) == 0
        fprintf("%i\n",j);
        splitName = split(fileList(1).name,"."); datetime = splitName{4};
        filename = sprintf(...
            "D:\\SHARP_data_v3\\hmi.sharp_cea_720s.%i.%s", ...
            sharp,datetime);
        trial = fitsread(filename+".Br.fits","image");
        if max(abs(trial(:))) < 1e4 && ...
                min(size(fitsread(filename + ".Br.fits","image"))) > 100
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
parpool('Processes',availableGPUs);
parfor newj = 1:sharpsn
    tic;

    j = sharps(newj);
    parprint(sprintf("j=%i/%i starting...\n",newj,sharpsn));

    Br = flipud(Brs{j})/1e3;
    Bt = flipud(Bts{j})/1e3;
    Bp = flipud(Bps{j})/1e3;
    % Build the nodeset
    numpoints = 1e5;

    % determine dimensions
    imageX = size(Br,1)/min(size(Br)); imageY = size(Br,2)/min(size(Br));
    imageZ = 2*max(imageX,imageY);
    
    % initialize x,y plane
    switch 2
        case 1
            window = 20;
            BrCropped = Br(1:floor(size(Br,1)/window)*window,1:floor(size(Br,2)/window)*window);
            supersmooth = sepblockfun(abs(BrCropped-mean(BrCropped(:))),[window,window],'max');
            [Xsm,Ysm] = ndgrid((0:size(supersmooth,1)-1)*window*imageX/(size(Br,1)-1), ...
                               (0:size(supersmooth,2)-1)*window*imageY/(size(Br,2)-1));
        case 2
            w = floor(min(size(Br))/40)*2+1;
            supersmooth = slideMaxFilter(abs(Br-mean(Br(:))),w);
            [Xsm,Ysm] = ndgrid(linspace(0,imageX,size(supersmooth,1)), ...
                               linspace(0,imageY,size(supersmooth,2)));
    end
    supersmooth = supersmooth/max(supersmooth(:));

    smoothfn = griddedInterpolant(Xsm,Ysm,supersmooth,'linear','nearest');
    omega = 1;
    rfn = @(xyz)((1-smoothfn(xyz(:,1),xyz(:,2)))*max(imageX,imageY)*1e-2+1.5e-2).*exp(omega*xyz(:,3)/imageZ);
    nodes = node_drop_3d ([0 imageX 0 imageY 0 imageZ], size(Br), numpoints, rfn);

    eps = 1e-4;
    
    index_z0 = find(nodes(:,3)<eps*imageZ);
    n0 = numel(index_z0);

    nodes = [nodes(index_z0,:)-[zeros(n0,2),rfn(nodes(index_z0,:))];nodes];%#ok<AGROW>
    index_gh = (1:n0)';

    index_z0 = index_z0 + n0;
    index_x0 = find(nodes(:,1)<eps*imageX);
    index_y0 = find(nodes(:,2)<eps*imageY);
    index_x1 = find(nodes(:,1)>imageX-eps*imageX);
    index_y1 = find(nodes(:,2)>imageY-eps*imageY);
    index_z1 = find(nodes(:,3)>imageZ-eps*exp(imageZ*omega));

    n = size(nodes,1);
    
    
    index_bd = unique([index_z0;index_z1;index_gh;index_x0;index_x1;index_y0;index_y1]);
    index_in = setdiff((1:n)',index_bd);
    index_re = setdiff((1:n)',index_gh);
    
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
    rns = zeros(sampleN,1);
    forcevec = 0*Bns;
    gamma = 1e-2;

    [X,Y] = ndgrid(linspace(0,imageX,size(Br,1)),linspace(0,imageY,size(Br,2)));
    Bx = griddedInterpolant(X,Y,-Bt);
    By = griddedInterpolant(X,Y,-Bp);
    Bz = griddedInterpolant(X,Y,Br);
    Bx0 = Bx(nodes(index_z0,1),nodes(index_z0,2));
    By0 = By(nodes(index_z0,1),nodes(index_z0,2));
    Bz0 = Bz(nodes(index_z0,1),nodes(index_z0,2));
    
    samplepresnode = net(haltonset(3),pressn);
    samplepresnode(:,1) = samplepresnode(:,1)*imageX;
    samplepresnode(:,2) = samplepresnode(:,2)*imageY;
    samplepresnode(:,3) = samplepresnode(:,3)*imageZ;
    [~,pDx,pDy,pDz] = RBF_find_global_weights( ...
                samplepresnode(:,1),samplepresnode(:,2),samplepresnode(:,3),...
                nodes(index_z0,1),nodes(index_z0,2),nodes(index_z0,3), ...
                3, 0 ...
               );
    [Q,R] = qr((pDx.*Bx0+pDy.*By0+pDz.*Bz0)');
    % [r,rd] = sort(abs(diag(R)));
    % sd = randperm(max(find(r>=1e-3,1),sampleN),sampleN);
    % samplepressure = Q(:,rd(sd));
    rd = find(abs(diag(R))<=1e-3);
    Qweights = sum((imageZ-samplepresnode(:,3)).*abs(Q(:,rd)),1)/pressn;
    [~,sd] = maxk(Qweights,sampleN);
    samplepressure = 0.5*Q(:,rd(sd));

    [Bpx,Bpy,Bpz] = potfield(nodes,index_z0,Bz0,64);
    [Bff,rff] = num_mhs(zeros(n,1),zeros(n,3),Bx0,By0,Bz0, n, ...
            nodes, Dx, Dy, Dz, Dxx, Dyy, Dzz, 1,...
            index_x0,index_x1,index_y0,index_y1,index_z0,index_z1,index_gh, ...
            gamma,[Bpx;Bpy;Bpz]);
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
            gamma,[Bpx;Bpy;Bpz]);
        rns(cpuj) = rs(end);

        forcevec(:,cpuj) = [presx;presy;presz];

        parprint(sprintf(" -%i - %i/%i done, r=%s\n",newj,cpuj,sampleN,sprintf(" %e ",rs)));
    end
    parprint(sprintf("j=%i/%i done in %f.\n",newj,sharpsn,toc));
    parsave(sprintf("D:\\MHS_solutions_v4\\sharp%i.mat", allsharps(j)), ...
            Bns([index_re;n+index_re;2*n+index_re],:), ...
            nodes(index_re,:), ...
            index_z0-n0,n-n0, ...
            rns, ...
            forcevec([index_re;n+index_re;2*n+index_re],:), ...
            Bff([index_re;n+index_re;2*n+index_re],:) ...
    );
end
%%
close all;
tri = delaunay(nodes(index_z0,1),nodes(index_z0,2));
for plotind = 0:6
    if plotind == 0
        Bn = Bff;
        fn = zeros(n,1);
    else
        Bn = Bns(:,plotind);
        fn = sqrt(forcevec(1:n,plotind).^2+forcevec(n+1:2*n,plotind).^2+forcevec(2*n+1:3*n,plotind));
    end
    threshold = 1e-3;
    % Plotting
    % mask = find(sqrt(Bn(1:n).^2+Bn(n+1:2*n).^2+Bn(2*n+1:3*n).^2)<threshold);
    % Bn([mask;n+mask;2*n+mask]) = 0;
    numX = scatteredInterpolant(nodes, Bn(1:n),'nearest');
    numY = scatteredInterpolant(nodes, Bn(n+1:2*n),'nearest');
    numZ = scatteredInterpolant(nodes, Bn(2*n+1:3*n),'nearest');
    numP = scatteredInterpolant(nodes, fn,'nearest');
    numB = scatteredInterpolant(nodes, sqrt(Bn(1:n).^2+Bn(n+1:2*n).^2+Bn(2*n+1:3*n).^2),'nearest');
    
    perdim = 100;
    [xq,yq,zq] = meshgrid(linspace(0,max(nodes(:,1)),perdim), ...
                          linspace(0,max(nodes(:,2)),perdim), ...
                          linspace(0,max(nodes(:,3)/2),perdim));
    Bxnq = numX(xq,yq,zq);
    Bynq = numY(xq,yq,zq);
    Bznq = numZ(xq,yq,zq);
    
    % Streamlines
    step = 4;
    startx = squeeze(xq(1:step:end,1:step:end,1));
    starty = squeeze(yq(1:step:end,1:step:end,1));
    startz = squeeze(zq(1:step:end,1:step:end,1));
    
    streamsForw = stream3(xq,yq,zq,Bxnq,Bynq,Bznq,startx(:),starty(:),startz(:));
    streamsBack = stream3(xq,yq,zq,-Bxnq,-Bynq,-Bznq,startx(:),starty(:),startz(:));
    fig = figure(plotind+1);
    ax = axes('Parent',fig);
    h1 = streamtube(streamsForw,0.01);
    hold on;
    h2 = streamtube(streamsBack,0.01);
    % h3 = slice(xq,yq,zq,Bznq,[],[],0);
    h3 = trisurf(tri,nodes(index_z0,1),nodes(index_z0,2),nodes(index_z0,3),Bn(2*n+index_z0));
    set(h3,'edgecolor','flat');

    for streamind=1:length(h1)
        set(h1(streamind),'AlphaData',numB(get(h1(streamind),'XData'),get(h1(streamind),'YData'),get(h1(streamind),'ZData')));
        set(h1(streamind),'FaceColor','black','FaceAlpha','interp','EdgeColor','none');
    end
    for streamind=1:length(h2)
        set(h2(streamind),'AlphaData',numB(get(h2(streamind),'XData'),get(h2(streamind),'YData'),get(h2(streamind),'ZData')));
        set(h2(streamind),'FaceColor','black','FaceAlpha','interp','EdgeColor','none');
    end
    view(3);
    set(gca,'Fontsize',16);
    axis('equal');
    view(ax,[-91.5 76.5]);
    xlabel('x'); ylabel('y'); zlabel('z');

    colormap('parula'); clim([-1 1]); 
    hold off;
end


%% functions
function parsave(fname, Bns,nodes,index_z0,n,rns,forcevec,Bff)
 save(fname,"Bns","nodes","index_z0","n","rns","forcevec","Bff");
end

function parprint(string)
    fprintf(string);
end
