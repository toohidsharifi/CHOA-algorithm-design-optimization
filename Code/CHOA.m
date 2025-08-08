clc;
clear;
close all;

%% Problem Definition

CostFunction = @(x)	funSphere(x);						% Cost Function for the minimization problem
nVar = 100;									% Number of Decision variables
VarSize = [1 nVar];								% Decision variables matrix size
VarMin = -1000;									% Lower bound of the Decision Variables
VarMax = 1000;									% Upper bound of the Decision Variables

%% CHOA Parameters

MaxIt = 1000;									% Maximum number of iterations
nPop = 50;									% Population size
nRank = 10;									% Rank size

Pcomb = 0.5;									% Combination Percentage
n_comb = 2*round(Pcomb*nPop/2);							% Number of offsprings

Pcomp = 0.5;									% Competition Percentage
n_comp = 2*round(Pcomp*nPop/2);							% Number of competitors

gamma=0.05;									% Coefficient for the Uniform Combination
%% Initialization

empty_individual.Position = [];
empty_individual.Cost = [];
empty_individual.rank = [];

pop = repmat(empty_individual, nPop, 1);

for i = 1:nPop

    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);				% Create Initial Idea System
    pop(i).Cost = CostFunction(pop(i).Position);				% Evaluate the Cultural Value of the System

end

Costs = [pop.Cost];
[Costs, SortOrder] = sort(Costs);
pop = pop(SortOrder);								% Sort the Idea System based on the Cultural Values

% Assign Ranks
A = nPop/nRank;									% Index
A1 = floor(A);									% Number of ideas in the rank
Prank = nPop - A1*(nRank-1);							% The Last Rank

S = 1;

for j = 1 : nRank-1

    for k = S : A1 + S - 1

	pop(k).rank = j;

    end

    S = k + 1;
end

for d = S: Prank+S-1

    pop(d).rank = nRank;

end

BestCosts = zeros(MaxIt, 1);

% Store Cost
WorstCost=pop(end).Cost;

BestSol = pop(1);

G = 0;										% Rank Size Flag

if nRank>1

    G = 1;

end

betaCombination=8;								% Selection Pressure of Combination
betaCompetition=3;								% Selection Pressure of Competition

%% Main Loop

for it = 1: MaxIt

    % Combination
    % Calculate Selection Probabilities

    PCombination=exp(-betaCombination*Costs/WorstCost);
    PCombination=PCombination/sum(PCombination);

    popc=repmat(empty_individual,n_comb/2,2);

    for k=1:n_comb/2

	% Select Parents Indices

	% Selecting the indices using the Roulette Wheel Selection Method

	i1Combination=RouletteWheelSelection(PCombination);

	i2Combination=RouletteWheelSelection(PCombination);

	if G == 1

	    while pop(i1Combination).rank == pop(i2Combination).rank

		i2Combination=RouletteWheelSelection(PCombination);

	    end

	end

	% Select Parents

	p1Combination=pop(i1Combination);

	p2Combination=pop(i2Combination);

	% Apply Combination

	[popc(k,1).Position, popc(k,2).Position]=...
	    Combination(p1Combination.Position,p2Combination.Position,...
	    gamma,VarMin,VarMax);

	% Evaluate Offsprings

	popc(k,1).Cost=CostFunction(popc(k,1).Position);

	popc(k,2).Cost=CostFunction(popc(k,2).Position);

    end

    popc=popc(:);

    % Competition

    % Calculate Selection Probabilities

    PCompetition=exp(-betaCompetition*Costs/WorstCost);
    PCompetition=PCompetition/sum(PCompetition);

    popb=repmat(empty_individual,n_comp/2,2);

    for k=1:n_comp/2

	% Select Parents Indices

	% Select the indices using the Roulette Wheel Selection

	i1Competition=RouletteWheelSelection(PCompetition);

	i2Competition=RouletteWheelSelection(PCompetition);

	if G == 1

	    while pop(i1Competition).rank == ...
		    pop(i2Competition).rank

		i2Competition=RouletteWheelSelection(PCompetition);

	    end

	end

	i1Competition=nPop+1- i1Competition;					% Making the inverse selection

	i2Competition=nPop+1- i2Competition;					% Making the inverse selection

	% Select Parents

	p1Competition=pop(i1Competition);

	p2Competition=pop(i2Competition);

	% Apply Competition

	[popb(k).Position]=...
	    Competition(p1Competition,p2Competition,BestSol,...
	    nVar,VarMin,VarMax);

	% Evaluate Competitor

	popb(k).Cost=CostFunction(popb(k).Position);

    end

    popb=popb(1:n_comp/2)';

    % Create Merged Population

    pop=[pop
	popc
	popb];									%#ok

    % Sort Population

    Costs=[pop.Cost];

    [Costs, SortOrder]=sort(Costs);

    pop=pop(SortOrder);

    % Selection

    pop=pop(1:nPop);

    Costs=Costs(1:nPop);

    % Assign rank

    S = 1;

    for j = 1 : nRank-1

	for k = S : A1 + S - 1

	    pop(k).rank = j;

	end

	S = k + 1;
    end

    for d = S: Prank+S-1

	pop(d).rank = nRank;

    end

    % Update Best Solution Ever Found

    BestSol=pop(1);

    % Update Best Cost Ever Found

    BestCosts(it)=BestSol.Cost;

    % Show Iteration Information

    disp(['Iteration = ' num2str(it), ', Best Cost = ' num2str(BestCosts(it))]);

end

%% Results

figure;

semilogy(1:MaxIt,BestCosts,'LineWidth',2);

xlabel('Iteration');

ylabel('Cost');

%% Functions:

function [y1 y2]=Combination(x1,x2,gamma,VarMin,VarMax)

    alpha=unifrnd(-gamma,1+gamma,size(x1));
    
    y1=alpha.*x1+(1-alpha).*x2;
    y2=alpha.*x2+(1-alpha).*x1;
    
    y1=max(y1,VarMin);
    y1=min(y1,VarMax);
    
    y2=max(y2,VarMin);
    y2=min(y2,VarMax);

end

function y=Competition(x1,x2,BestSol,nVar,VarMin,VarMax)
     
	pop = [x1
	       x2];
	Costs = [pop.Cost];
        [~, SortOrder] = sort(Costs);
        pop = pop(SortOrder);

	alpha = rand;
	beta = unifrnd(0, 0.25, 1, 1);

	a = 1:nVar;

	pop(2).Position(a) = alpha*pop(1).Position(a) + beta*BestSol.Position(a);
        
	y = pop(2).Position;       
        
        y(a)=max(y(a),VarMin);
        y(a)=min(y(a),VarMax);      

end

function i=RouletteWheelSelectionCombination(P)

    r=rand;
    
    c=cumsum(P);
    
    i=find(r<=c,1,'first');

end

function [p,dp] = funSphere(xx)

%evaluation and derivatives
pX=xx.^2;
%
p=sum(pX);

if nargout==2
    %
    dp=2*xx;
end
end

Create code folder and add main script
