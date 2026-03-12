function [DD,CI] = RFC_getRFDist(Xdata,rfc,param,CI)
% [DD,CI] = RFC_getRFDist(Xdata,rfc,param,CI)
% compute distance from Random Forests
%
% Random Forest Clustering library
% (c) Manuele Bicego 2021
%
% Xdata: [nxd]: n objects in a d-dimensional space
% rfc is the trained forest
% param.dist indicates the distance
%   1 -> Shi et al., J. Comput. Graph. Statist., 15(1), 118-138, 2006.
%   2 -> v2 of Zhu et al CVPR, 1450–1457, 2014
%   3 -> v3 of Zhu et al CVPR, 1450–1457, 2014
%   4 -> Ting et al, SIGKDD, 1205–1214, 2016
%   5 -> RatioRF
%   6 -> Ayral et al, Ayral et al, Data Mining & Knowledge Discovery 2020

if nargin < 4
    CI = RFC_getRFClusInfo(Xdata,rfc,param);
end
nT = length(CI);
n = size(Xdata,1);
% fprintf('Computing Distance\n');
switch param.dist
    case 1
        %   1 -> Shi et al., J. Comput. Graph. Statist., 15(1), 118-138, 2006.
        X = [CI.leafid];
        DD = sqrt(squareform(pdist(X,'hamming')));
    case 2
        %   2 -> v2 of Zhu et al CVPR, 1450–1457, 2014
        % tic
        DD = zeros(n);
        for t = 1:nT
            if (mod(t,10) == 0) || (t == 1) || (t == param.ntrees), if param.verbose, fprintf('Tree %d/%d\n',t,param.ntrees); end; end
            xDD = zeros(n);
            beta = CI(t).beta;
            d = sum(beta);
            beta(:,d==0) = [];
            lnp = sum(beta,2);
            for i = 1:n
                bi = (beta & repmat(beta(i,:),[n,1]));
                xd = sum(bi,2)-1;
                xd = xd./(max(lnp(i),lnp)-1);
                xDD(:,i) = xd;
            end
            DD = DD+xDD;
        end
        DD = 1-DD./nT;
        % OPTIMIZATION trials: NOTE: apparently they are worsening the performances:
        %     It seems it is better to make
        %     two nested loops instead of vreating bigger matrices and work
        %     without loops: problem of memory?
        %     same if using cellfun
        % VERSION 1. version with cellfun: apparently no speed improvement
        %         a = toc;
        %         tic
        %         beta = {CI.beta};
        %         xd = cellfun(@RFC_zhu2,beta,'UniformOutput',false);
        %         xdd = reshape([xd{:}],[n n nT]);
        %         DD2 = 1-mean(xdd,3);
        %         sum(sum(abs(DD2-DD)))
        %         b = toc;
        %         [a b]
        
        %  VERSION 2: optimized with repmat
        %         %   2 -> v2 of Zhu et al CVPR, 1450–1457, 2014
        %         % tic
        %         DD = zeros(n);
        %         Abeta = [CI.beta];
        %         mx = zeros(nT,1);
        %         Alnp = zeros(n,nT);
        %         for t = 1:nT
        %             beta = CI(t).beta;
        %             Alnp(:,t) = sum(beta,2);
        %             mx(t) = size(beta,2);
        %         end
        %         for i = 1:n
        %             bi = (Abeta & repmat(Abeta(i,:),[n,1]));
        %             nf = repmat(Alnp(i,:),[n,1]);
        %             nf2 = max(nf,Alnp)-1;
        %             nf3 = repelem(nf2,ones(n,1),mx);
        %             bi = bi./nf3;
        %             DD(:,i) = sum(bi,2) - sum(1./nf2,2);
        %         end
        %         DD = 1-DD./nT;
        
        % VERSION 3: optimized with cellfun
        %         %   2 -> v2 of Zhu et al CVPR, 1450–1457, 2014
        %         % tic
        %         DD = zeros(n);
        %         Cbeta = {CI.beta};
        %         Abeta = [CI.beta];
        %         Cmx = cellfun(@size,Cbeta,num2cell(2*ones(1,nT)),'UniformOutput',false);
        %         Amx = [Cmx{:}];
        %         Clnp =  cellfun(@sum,Cbeta,num2cell(2*ones(1,nT)),'UniformOutput',false);
        %         Alnp = [Clnp{:}];
        %         for i = 1:n
        %             bi = (Abeta & repmat(Abeta(i,:),[n,1]));
        %             Cnormfact = cellfun(@max,Clnp,num2cell(Alnp(i,:)),'UniformOutput',false);
        %             Cnormfact = cellfun(@minus,Cnormfact,num2cell(ones(1,nT)),'UniformOutput',false);
        %             b = mat2cell([ones(nT,1) Amx'],ones(nT,1));
        %             nor = cellfun(@repmat,Cnormfact,b','UniformOutput',false);
        %             Anor = [nor{:}];
        %             bi = bi./Anor;
        %             Anormfact = 1./[Cnormfact{:}];
        %             DD(:,i) = sum(bi,2)-sum(Anormfact,2);
        %         end
        %         DD = 1-DD./nT;
    case 3
        %   3 -> v3 of Zhu et al CVPR, 1450–1457, 2014
        DD = zeros(n);
        for t = 1:nT
            if (mod(t,10) == 0) || (t == 1) || (t == param.ntrees), if param.verbose, fprintf('Tree %d/%d\n',t,param.ntrees); end; end
            xDD = zeros(n);
            beta = CI(t).beta;
            d = sum(beta);
            beta(:,d==0) = [];
            no = sum(beta);
            we = 1./(no + (no==0));
            % weighted beta
            wbeta = beta.*repmat(we,[n,1]);
            lnp = sum(wbeta(:,2:end),2);
            %lnp = sum(wbeta,2);
            for i = 1:n
                bi = (beta & repmat(beta(i,:),[n,1]));
                wbi = wbeta .* bi;
                xd = sum(wbi(:,2:end),2);
                %xd = sum(wbi,2);
                xd = xd./(max(lnp(i),lnp));
                xDD(:,i) = xd;
            end
            DD = DD+xDD;
        end
        DD = 1-DD./nT;
    case {4,6}
        %   4 -> Ting et al, SIGKDD, 1205–1214, 2016
        %   6 -> Ayral et al, Ayral et al, Data Mining & Knowledge Discovery 2020
        %   (basically, the difference is in taking geometric mean instead of arithmetic mean)    
        DD = zeros(n);
        for t = 1:nT
            if (mod(t,10) == 0) || (t == 1) || (t == param.ntrees), if param.verbose, fprintf('Tree %d/%d\n',t,param.ntrees); end; end
            xDD = zeros(n);
            beta = CI(t).beta;
            lev = CI(t).lev;
            nb = size(beta,2);
            mass = sum(beta); % number of points passing from each node
            
            for i = 1:n
                % for j = 1:n
                %    bi = beta(i,:) & beta(j,:);
                %    ndx = find(bi == 1);
                %    c = lev(ndx);
                %    [~,r] = max(c);
                %    m(i,j) = mass(ndx(r));
                %    % idn(n1)>idn(n2) -> depth(n1)>depth(n2)
                %    m2(i,j) = mass(ndx(end));
                % end
                % compact version
                bi = (beta & repmat(beta(i,:),[n,1]));
                bi = bi.*(repmat([1:nb],n,1));
                [~,r] = max(bi,[],2);
                xDD(i,:) = mass(r);
            end
            if param.dist == 4
                 %   4 -> Ting et al, SIGKDD, 1205–1214, 2016
                DD = DD+xDD./n;
            else
                %   6 -> Ayral et al, Ayral et al, Data Mining & Knowledge Discovery 2020
                logxDD = log(xDD./n);
                DD = DD+logxDD;
            end
                
        end
        DD = DD./nT;
        if param.dist == 6
            % scale between 0 and 1
            DD = (DD - min(DD(:)))/(max(DD(:))-min(DD(:)));
        end
        
    case 5 % RatioRF
        DD = zeros(n);
        for t = 1:nT
            if (mod(t,10) == 0) || (t == 1) || (t == param.ntrees), if param.verbose, fprintf('Tree %d/%d\n',t,param.ntrees); end; end
            xDD = zeros(n);
            xDD2 = zeros(n);
            xDD3 = zeros(n);
            xcx = zeros(n);
            xlambda = zeros(n);
            beta = CI(t).beta;
            b = CI(t).b;
            dep = CI(t).depObj;
            % removing leaves
            d = (sum(b)==0);
            beta(:,d) = [];
            b(:,d) = [];
            dep = dep - 1;
            nb = size(beta,2);
            for i = 1:n
                %for j = 1:n
                %    alltest = beta(i,:) | beta(j,:);
                %    common(i,j) = sum(b(i,alltest) == b(j,alltest));
                %    cx(i,j) = sum(b(i,beta(i,:)) == b(j,beta(i,:)));
                %    cy(i,j) = sum(b(i,beta(j,:)) == b(j,beta(j,:)));
                %    % OLD VER lambda(i,j) = sum(beta(i,:) & beta(j,:))-1;
                %    lambda(i,j) = sum(beta(i,:) & beta(j,:));
                %    den = dep(i) + dep(j) - lambda(i,j);
                %    xDD(i,j) = common(i,j)/den;
                %    xDD2(i,j) = (cx(i,j)+cy(i,j)-lambda(i,j))/den;
                %end
                r = sum(repmat(b(i,beta(i,:)),n,1) == b(:,beta(i,:)),2);
                xcx(i,:) = r';
                xlambda(i,:) = sum((repmat(beta(i,:),n,1) & beta),2);
                %%% non funzia 
                %XCx = sum(repmat(b(i,beta(i,:)),n,1) == b(:,beta(i,:)),2);
                %XCy = sum(b(i,beta)) == b(:,beta),2);
                %Xlambda = sum((repmat(beta(i,:),n,1) & beta),2);
                %den = dep(i) + dep - Xlambda;
                %xDD3(i,:) = (XCx + XCy - Xlambda)./den;
            end
            den = repmat(dep',n,1) + repmat(dep,1,n) - xlambda;
            li = CI(t).leafid;
            % the correction (li ~= li') considers mismatch between edges
            % (theory) and nodes (implementation)
            xDD3 = (xcx+xcx'-xlambda +(li ~= li'))./den;
            DD = DD+xDD3;
        end
        DD = DD./nT;
        DD = sqrt(1-DD);
end


% % internal functions
%
% function xDD = RFC_zhu2(beta)
%
% n = size(beta,1);
% xDD = zeros(n);
% lnp = sum(beta,2);
% for i = 1:n
%     bi = (beta & repmat(beta(i,:),[n,1]));
%     xd = sum(bi,2)-1;
%     xd = xd./(max(lnp(i),lnp)-1);
%     xDD(:,i) = xd;
% end
