% Script to produce a set of circular phantoms
% Andreas Hauptmann, 2017



%% Create Matrix
createMat=true;
recSize=64;


Nx = recSize;       % number of grid points in the x (row) direction
Ny = recSize;       % number of grid points in the y (column) direction
dx = 0.1e-3;        % grid point spacing in the x direction [m]
dy = 0.1e-3;        % grid point spacing in the y direction [m]
kgrid = kWaveGrid(Nx, dx, Ny, dy);

% define the properties of the propagation medium
c = 1500;           % sound speed [m/s]
% set the sound speed
medium.sound_speed = c;

% set the time stepping
dt = dx/c;
kgrid.t_array = (0:(2*Ny-1)) * (dt/2);

% define the sensors on the plane y=0
sensor.mask = zeros(Nx,Ny);
sensor.mask(1,:) = 1;


% kgrid.makeTime(medium.sound_speed);

if(createMat)
    %Initialise system matrix
    A=zeros(Nx*Ny*2,Nx*Ny);

    for iii=1:recSize*recSize
    
        

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % create initial pressure distribution using makeDisc
        p0=zeros(Nx,Ny);
        p0(iii)=1;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % k-Wave simulation

        % set the initial condition (photoacoustic source term)
        source.p0 = p0;

        % run the k-Wave simulation
        input_args = {'PMLInside', false, 'PMLSize', 8, 'Smooth', false, ...
            'PlotPML', false, 'PlotSim', false};

        % run the simulation
        sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
        
        
        
        %Record column iii
        A(:,iii)=sensor_data(:);


        
        display(['Iteration: ' num2str(iii) ' of ' num2str(Nx*Ny)])
        
    end

end
%%

% We can treshold the matrix to remove small entries and make it more
% sparse 
 
for thresh = 6
    Athresh=A;
    
    Athresh(abs(Athresh)<10^(-thresh))=0;
   
end


% %Transpose first
 A=A'; Athresh=Athresh';
%save for python
save('forwMat20.mat','-v7.3','A','Athresh')




