function [nodes] = electrostatic_repulsion(nodes,k,powerfn,impulse,boundaries,repeat)
%electrostatic_repulsion is meant to rearrange nodes so that they are more
% clustered where powerfn is low and less clustered where powerfn is high.
% nodes : NxD array of starting node locations
% k : how many nearest neighbors to consider when repelling - recommended
% to be at least 3^D-1.
% @powerfn : a non-negative elementwise-evaluated function that
% determines the clustering at that point in space. lower -> more
% clustered.
% impulse : how hard to do the electrostatic push.
% boundaries : [xmin xmax ymin ymax]
for k = 1:repeat
    idx = knnsearch(nodes,nodes,'k',k+1);
    idx = idx(:,2:end);
    switch size(nodes,2)
        case 1
            power = powerfn(nodes);
        case 2
            power = powerfn(nodes(:,1),nodes(:,2));
        case 3
            power = powerfn(nodes(:,1),nodes(:,2),nodes(:,3));
        case 4
            power = powerfn(nodes(:,1),nodes(:,2),nodes(:,3),nodes(:,4));
    end
    pushnodes = permute(reshape(nodes(idx,:),[],k,size(nodes,2)),[1,3,2]);
    distances = squeeze(sum((nodes-pushnodes).^2,2));
    pushPower = (power(idx)+1) ./ distances;
    push = (nodes-pushnodes).*reshape(pushPower,[],1,k);
    
    nodes = nodes + squeeze(sum(push,3))*impulse;
    
    nodes(nodes(:,1)<boundaries(1),1) = 2*boundaries(1)-nodes(nodes(:,1)<boundaries(1),1);
    nodes(nodes(:,1)>boundaries(2),1) = 2*boundaries(2)-nodes(nodes(:,1)>boundaries(2),1);
    if size(nodes,2) > 1
        nodes(nodes(:,2)<boundaries(3),2) = 2*boundaries(3)-nodes(nodes(:,2)<boundaries(3),2);
        nodes(nodes(:,2)>boundaries(4),2) = 2*boundaries(4)-nodes(nodes(:,2)>boundaries(4),2);
    end
    if size(nodes,2) > 2
        nodes(nodes(:,3)<boundaries(5),3) = 2*boundaries(5)-nodes(nodes(:,3)<boundaries(5),3);
        nodes(nodes(:,3)>boundaries(6),3) = 2*boundaries(6)-nodes(nodes(:,3)>boundaries(6),3);
    end
end

nodes(nodes(:,1)<boundaries(1),1) = boundaries(1);
nodes(nodes(:,1)>boundaries(2),1) = boundaries(2);
if size(nodes,2) > 1
    nodes(nodes(:,2)<boundaries(3),2) = boundaries(3);
    nodes(nodes(:,2)>boundaries(4),2) = boundaries(4);
end
if size(nodes,2) > 2
    nodes(nodes(:,3)<boundaries(5),3) = boundaries(5);
    nodes(nodes(:,3)>boundaries(6),3) = boundaries(6);
end
    

end