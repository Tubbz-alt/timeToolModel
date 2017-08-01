#!/reg/g/psdm/sw/releases/ana-current/arch/x86_64-rhel7-gcc48-opt/bin/python

import psana;
import numpy as np;
#import matplotlib.pyplot as plt;
import TimeTool;
import argparse;

# XPP 12mu water runs: 106 = 10%, 107 = 1%, 109 = 50%, 110 = 100%
# XPP 20mu water runs: 99 = .8%, 100 = .1%, 101 = 10%, 
# XPP diamond runs: 17 = 30%, 20 = 50%, 21 = 100%, 44 = 100%+slottedfoil10fs
# XPP SiN 1mu runs: 54 = 5%, 58 = 50%, 59 = 100%, 64 = 100%
# XPP SiN 2mu runs: 102 = 1%, 103 = 10%, 104 = 5%
limstr="Signal vert lims:\tr17-64 = (205:225)\tr99-r104 = (105:110)\tr106-r110 = (95:100)"
helpstr = 'Compute Matrix and shot sorting\tfor XPP interferometric method intensity scans.\t' + limstr;
parser = argparse.ArgumentParser(description=helpstr);#'Compute Matrix and shot sorting\n\tfor XPP interferometric method intensity scans');
parser.add_argument('-r','--run',dest='runstr', type=str, help='run number', default='52');
parser.add_argument('-d','--dirname',dest='dirname',type=str, help='directory name for data',default='data/');
parser.add_argument('-l','--llim',dest='llim',type=int,help='signal vrange lower limit (inclusive)',default=int(205));
parser.add_argument('-u','--ulim',dest='ulim',type=int,help='signal vrange upper limit (exclusive)',default=int(225));
parser.add_argument('-n','--skipshots',dest='skipshots',type=int,help='skip n shots',default=int(20));
parser.add_argument('-s','--skipsteps',dest='skipsteps',type=int,help='skip n steps',default=int(2));
parser.add_argument('-a','--attenuation',dest='atten',type=float,help='attenuation (actually transmission e.g. .1 xrays on sample = .1',default=float(1));

args = parser.parse_args();

headstr = '#';


def i2lam(i):
        lset=550;
        nmPi=0.217;
        seterr=1.0051;
        return nmPi*i + seterr*lset - 110.072;
#'i' is the pixel index [ 0 .. 1023 ] and the wavelength [nm]
# run 31 is 1 micron SiN and shows strong etalon
#runstr = str('29');
#expstr = str('amo11816');
#runstr = str('52');
runstr = args.runstr;
expstr = str('xppl3816');
dsourcestr = 'exp=' + expstr + ':run=' + runstr;
print(dsourcestr)

# setting up TimeTool
bykick = int(162);
ttOptions = TimeTool.AnalyzeOptions(get_key='opal_1', eventcode_nobeam = bykick, sig_roi_x = '1 1022', sig_roi_y = '350 375',sb_roi_x = '1 1022',sb_roi_y = '150 175');
ttAnalyze = TimeTool.PyAnalyze(ttOptions);
ttSign = float(-1); # for XPP we need to subtract the TimeTool delay

ds = psana.DataSource(dsourcestr,module=ttAnalyze);
print(psana.DetNames('detectors'));
#det = psana.Detector('OPAL1') # for AMO
evr = psana.Detector('NoDetector.0:Evr.0')
det = psana.Detector('opal_0'); #'XppEndStation.0:Opal1000.0') # for XPP
TTdet = psana.Detector('opal_1'); #This is the TimeTool camera I think
cd = psana.Detector('ControlData')
EBdet = psana.Detector('EBeam'); #This I hope is the BLD data
GDdet = psana.Detector('FEEGasDetEnergy'); #This I hope is the FEE Gas Detector readings
skipshots = args.skipshots;
skipsteps = args.skipsteps;
num = 0.0;
y_init = 0;
y_final = 0;

#vwin = (490,495); #AMO
#vwin = (105,115); #XPP
#vwin = (104,114); #XPP
vwin = (args.llim,args.ulim);
num = vwin[1]-vwin[0];

printsample = True;
ratio = .1;
subref = False;

#delayscale=float(1e3); #AMO
delayscale=float(1e12); #XPP


'''The third edition: take the average of each step and convert both axes to the right units. '''
for run in ds.runs():
	reference_img = np.zeros(1024);
	nrefshots = int(0);
	for nstep,step in enumerate(run.steps()):
		if nstep%skipsteps==0:
			pvList = cd().pvControls();
			for pv in pvList:
				if y_init == 0:
					y_init = pv.value()
				y_final = pv.value()	
				print('Step', nstep, 'name/value',pv.name(),pv.value());
			for nevent,evt in enumerate(step.events()):
				ttResults = ttAnalyze.process(evt);
				if (printsample and nevent%500==0):
					img = det.image(evt);
					filename = args.dirname
					filename += '/images' + expstr + '_r' + runstr + '_image%d.step%d.dat' % (nevent,nstep);
					headstr='# sample image'
					printsample = False;
					filename = args.dirname
					filename += '/images' + expstr + '_r' + runstr + '_TTimage%d.step%d.dat' % (nevent,nstep);
					img = TTdet.image(evt);
					headstr='# sample image TimeTool camera'
					if ttResults!=None:
						print(ttResults.position_time());

				if subref:
					ec = evr.eventCodes(evt)
					if bykick in ec: 
						img = det.image(evt);
						if (img is None):
							continue;
						if nrefshots==0:
							reference_img = np.sum(img[vwin[0]:vwin[1],:],axis=0)/num;
						else:
							reference_img *= (1.-ratio);
							reference_img += (ratio)*np.sum(img[vwin[0]:vwin[1],:],axis=0)/num;
						nrefshots += 1;

				if nevent%skipshots == 0:
					lineout = np.zeros(1024,dtype=float);


					print('Fetching event number',nevent);
    					img = det.image(evt);  #img is not areal image, it is a matrix
 					if (img is None):
						continue;

					try:
						lineout = (np.sum(img[vwin[0]:vwin[1],:],axis=0)/num) ;
						tt_data = (ttResults.position_pixel(),ttResults.amplitude(),ttResults.position_time(),ttResults.position_fwhm());
						dd_data = (nstep,delayscale*y_final);#,delayscale*y_final+ttSign*1e-3*ttResults.position_time());
						eb_data = ();
						gd_data = ();
					except:
						continue;
					try:
						if subref:
							lineout = lineout - reference_img[:len(lineout)];
						R = np.row_stack((R,lineout));#it stacks from top to bottom
						D = np.row_stack((D,dd_data));
						T = np.row_stack((T,tt_data));
						E = np.row_stack((E,eb_data));
						G = np.row_stack((G,gd_data));
					except NameError:
						if subref:
							lineout = lineout - reference_img[:len(lineout)];
						R = lineout;
						D = dd_data;
						T = tt_data;
						E = eb_data;
						G = gd_data;

#for plot
y_dim = int(np.shape(R)[0]);
x_dim = int(np.shape(R)[1]);
lam = i2lam(np.arange(x_dim,dtype=float));
#delay = delayscale*np.linspace(y_init,y_init,y_dim,dtype=float);

sorted_inds = np.argsort(D[:,1]);

if subref:
	runstr += "_refsub";



#filename = args.dirname;#'data_tmp/';
#filename+=expstr + '_r' + runstr + '_matrix_sorted.dat';
#headstr='# sorted matrix';
#np.savetxt(filename,R[sorted_inds,:],fmt='%.6e');


filename = args.dirname;#'data_tmp/';
headstr='# index ordered to sort the raw matrix';
filename+=expstr + '_r' + runstr + '_inds_sorted.dat';
np.savetxt(filename,sorted_inds,fmt='%.6e',header=headstr);


filename = args.dirname;#'data_tmp/';
filename+=expstr + '_r' + runstr + '_matrix.dat';
np.savetxt(filename,R,fmt='%.6e');
filename = args.dirname;#'data_tmp/';
filename+=expstr + '_r' + runstr + '_delays.dat';
np.savetxt(filename,D,fmt='%.6e');
filename = args.dirname;#'data_tmp/';
filename+=expstr + '_r' + runstr + '_eb.dat';
np.savetxt(filename,E,fmt='%.6e');
filename = args.dirname;#'data_tmp/';
filename+=expstr + '_r' + runstr + '_tt.dat';
np.savetxt(filename,T,fmt='%.6e');
filename = args.dirname;#'data_tmp/';
filename+=expstr + '_r' + runstr + '_wavelegths.dat';
np.savetxt(filename,lam,fmt='%.6e');

print('Done saving');

#plt.imshow(R,origin = 'lower',extent = [lam[0],lam[-1],delay[0],delay[-1]],aspect = 'auto')
#tick_locs_x = np.linspace(450,650,5)
#plt.xticks(tick_locs_x)

#tick_locs_y = np.linspace(delay[0],delay[-1],6)
#plt.yticks(tick_locs_y)

#plt.hot()
#plt.colorbar()
#plt.show()				
print('Done.');
