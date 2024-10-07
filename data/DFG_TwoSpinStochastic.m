clc
clear all

%% Specify spin system parameters
sys.isotopes={'19F','19F'};
sys.magnet=9.3933;

inter.zeeman.scalar={-113.8796 -129.8002};   
inter.coupling.scalar=cell(2,2);
inter.coupling.scalar{1,2}=238.0633;           
inter.coordinates={[-0.0551   -1.2087   -1.6523]; [-0.8604   -2.3200   -0.0624]};
inter.relaxation={'redfield','t1_t2'};
inter.r1_rates={0 0};
inter.r2_rates=34.0359*{1 1};
inter.equilibrium='dibari';
inter.temperature=310;
inter.rlx_keep='secular';
inter.tau_c={0.5255e-9};

bas.formalism='sphten-liouv';
bas.approximation='none';

sys.enable={'gpu'};

spin_system=create(sys,inter);
spin_system=basis(spin_system,bas);

%% Plot 1-D spectra
parameters.offset=-1000;
parameters.sweep=10000;
parameters.npoints=2048;
parameters.zerofill=4096;
parameters.spins={'19F'};
parameters.axis_units='ppm';
parameters.rho0=state(spin_system,'L+','19F');
parameters.coil=state(spin_system,'L+','19F');

fid=liquid(spin_system,@acquire,parameters,'nmr');
fid=apodization(fid,'gaussian-1d',10);
spectrum=fftshift(fft(fid,parameters.zerofill));
figure(); plot_1d(spin_system,real(spectrum),parameters);

%% Stochastic trajectory

H0=hamiltonian(assume(spin_system,'nmr'));
R=relaxation(spin_system);

Fx=operator(spin_system,'Lx','19F');
Fy=operator(spin_system,'Ly','19F');

coils=[state(spin_system,'Lx','19F')  state(spin_system,'Ly','19F')];

H_lab=hamiltonian(assume(spin_system,'labframe'),'left');
rho=equilibrium(spin_system,H_lab);

dt=1e-5; 
sig_omega=2*pi*100;
nsteps=1e6;
ux=sig_omega*randn(nsteps,1);
uy=sig_omega*randn(nsteps,1);

L0=gpuArray(H0+1i*R); 
Fx=gpuArray(Fx); Fy=gpuArray(Fy);
rho=gpuArray(rho); coils=gpuArray(coils);

tic
fids=zeros(2,nsteps);
for n=1:nsteps
    fids(:,n)=gather(coils'*rho);
    G=L0+ux(n)*Fx+uy(n)*Fy;
    rho=step(spin_system,G,rho,dt);
end
toc



figure(); scale_figure([2.0 3.0])
time_axis=linspace(0,nsteps*dt,nsteps);
subplot(2,2,1); plot(time_axis,ux/(2*pi)); ktitle('controls');
axis tight; kgrid; klegend({'Fx'},'Location','NorthEast');
kxlabel('time, seconds'); kylabel('nut. freq., Hz');
subplot(2,2,3); plot(time_axis,uy/(2*pi)); 
axis tight; kgrid; klegend({'Fy'},'Location','NorthEast');

time_axis=linspace(0,nsteps*dt,nsteps);
subplot(2,2,2); plot(time_axis,fids(1,:)); ktitle('observables');
axis tight; kgrid; klegend({'Fx'},'Location','NorthEast');
kxlabel('time, seconds'); kylabel('expect. value');
subplot(2,2,4); plot(time_axis,fids(2,:));
axis tight; kgrid; klegend({'Fy'},'Location','NorthEast');
kxlabel('time, seconds'); kylabel('expect. value');


p.fids=fids;
p.ux=ux;
p.uy=uy;

save DFG_TwoSpinStochasticData.mat p















