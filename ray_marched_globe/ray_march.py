import numpy as np
import cv2, time, sys
import torch, torch.nn.functional as F
from .utils import *

BILINEAR_TEXTURE = True
LIVE_CONTROLS = True

def lookAt(eye, center, up0):
    f = -(eye - center); f = f / np.linalg.norm(f)
    l = np.cross(up0, f); l = l / np.linalg.norm(l)
    u = np.cross(f, l); u = u / np.linalg.norm(u)
    out = np.eye(4, dtype=f.dtype)
    out[0, :3] = l
    out[1, :3] = u
    out[2, :3] = f
    out[:3, 3] = -out[:3,:3] @ eye
    return out

class RayMarchingViz():
    def __init__(self, cfg):
        #cfg.setdefault('resolution', 512//1)
        cfg.setdefault('resolution', 756)
        cfg.setdefault('fov', np.pi/3)
        cfg.setdefault('iters', 18)
        self.eye = torch.FloatTensor([-3,0,0]).cuda()
        self.R = torch.eye(3).flip(0).cuda()
        self.iters = cfg['iters']
        self.w = cfg['resolution']
        self.h = cfg['resolution']
        self.u = np.tan(cfg['fov']/2)
        self.v = np.tan(cfg['fov']/2)
        self.t0 = time.time()

        #self.axes = torch.FloatTensor([[1,1,1.597]]).cuda()
        self.axes = torch.FloatTensor([[1,1,1.001]]).cuda()

        self.worldImg = torch.from_numpy(cv2.imread('./res/world.wm.jpg'))\
                .cuda().float().div_(255)
        #for _ in range(1): self.worldImg = F.avg_pool2d(self.worldImg.permute(2,0,1).unsqueeze_(0),2,2)[0].permute(1,2,0)
        #for _ in range(4): self.worldImg = F.avg_pool2d(self.worldImg.permute(2,0,1).unsqueeze_(0),2,2)[0].permute(1,2,0)
        if BILINEAR_TEXTURE:
            self.worldImg = self.worldImg.permute(2,0,1).unsqueeze_(0)
        self.llh = np.array((0,0,3.))

        self.x_y_deriv = torch.nn.Conv2d(1,2,3,padding=1,bias=False).cuda()
        self.x_y_deriv.weight.data.copy_(torch.FloatTensor((
            0,0,0, -1,0,1, 0,0,0,
            0,-1,0, 0,0,0, 0,1,0)).cuda().view(2,1,3,3))


    def update(self):
        t = (time.time() - self.t0) / 2
        self.eye = torch.FloatTensor([-np.cos(t), -np.sin(t), np.cos(t/2.1)]).cuda() * (1.6 + np.sin(t/2.23)*.5)
        self.R = torch.from_numpy(lookAt(
            self.eye.cpu().numpy(),
            np.zeros(3),
            np.array((0,0,1))
        )[:3,:3].astype(np.float32)).cuda().T
        print(' - at',self.eye)
        print(' - R:\n',self.R)
    def updateWithControls(self, dx,dy,dz,shift):
        speed = (.01 + .05*abs(self.llh[-1]))
        d = np.array((dx,dy,dz),dtype=np.float64) * speed
        self.llh += d
        lng,lat,alt = self.llh
        self.eye = torch.FloatTensor([np.cos(lat)*np.cos(lng), np.cos(lat)*np.sin(lng), np.sin(lat)]).cuda() * alt
        self.R = torch.from_numpy(lookAt(
            self.eye.cpu().numpy(),
            np.zeros(3),
            np.array((0,0,1))
        )[:3,:3].astype(np.float32)).cuda().T



    def render(self):
        with torch.no_grad():
            dev = torch.device('cuda')

            if not hasattr(self, 'uvws'):
                self.uvws = uvws = torch.stack(torch.meshgrid(
                    torch.linspace(-self.u,self.u, self.w, device=dev),
                    -torch.linspace(-self.v,self.v, self.h, device=dev),
                    torch.ones(1, device=dev)),-1).reshape(-1,3)
            else:
                uvws = self.uvws
            N = uvws.size(0)

            depths = torch.ones((N,1), device=dev) * 1e-6

            rays_ = uvws @ self.R.T

            for ii in range(self.iters):
                #rays = (uvws * depths) @ self.R.T
                rays = rays_ * depths
                dists = self.iso(self.eye, rays).view(-1,1).clamp_(-100,100)
                depths = depths.add_(dists)

            depths.masked_fill_(abs(depths)>14, 0)

            pts = self.eye + rays_ * depths

            if False:
                d = 1. - (depths - depths.min()) / (depths.max() - depths.min())
                #d = 1.3 - (depths / 3)
                d.masked_fill_(depths==0,0)
                d = d.view(self.h,self.w,-1).clamp_(0,1)
            else:
                # Of course you can send more rays and get better normals,
                # but screen space normals are faster.
                d = depths.view(1,1,self.h,self.w)
                screen_normal_xy = self.x_y_deriv(d)[0].permute(1,2,0)
                #screen_normal_z = torch.sqrt(1 - (screen_normal_xy**2).sum(-1))
                screen_normal_z = 1. - torch.sqrt((screen_normal_xy**2).sum(-1))
                d = screen_normal_z.clamp(0,1).unsqueeze_(-1)
                d = (d).pow_(16)
                d.masked_fill_(depths.view(self.h,self.w,1)==0, 0)

            #c = self.eye + rays_*depths
            #c = c.view(self.h,self.w,3).mul_(20).cos_().sqrt_()
            c = ecef_to_geodetic(pts)
            if BILINEAR_TEXTURE:
                c = F.grid_sample(
                        self.worldImg,
                        ((geodetic_to_unit_wm(c).view(self.h,self.w,2).unsqueeze_(0))))[0].permute(1,2,0)
            else:
                c = ((geodetic_to_unit_wm(c) * .5 + .5) * self.worldImg.size(0)).clamp(0,self.worldImg.size(0)-1).long()
                c = self.worldImg[c[:,1], c[:,0]].view(self.h,self.w,-1)

            color = c * d
            #color = c + (1-d)

            color = color.permute(1,0,2)
            color = color.cpu().numpy()
            cv2.imshow('RayMarching', color)
            cv2.waitKey(1)


    def iso(self, t, pts):
        pts_ = t + pts
        #return (pts_).norm(dim=1) - (1+self.noise(pts_)/2)
        #return (pts_.mul_(self.axes)).norm(dim=1).sub_(1)
        return (pts_.mul_(self.axes)).norm(dim=1).sub_(1 + self.hills(pts_))

    def hills(self, x):
        x = x / x.norm(dim=1,keepdim=True)
        #a = x[:,0].acos()
        #b = x[:,1].acos()
        return torch.sin(torch.cos(x[:,1] * 60) + x[:,0] * 60) * .03 + .01 \
             + torch.cos(x[:,1] * 8) * .09 + .01 \

    # https://iquilezles.org/www/articles/voronoise/voronoise.htm
    def noise(self, x):
        x = x * 5
        p = x.floor()
        f = x - p

        uu = 1
        v = 1

        k = 1 + 63 * (1-v)**4
        if not hasattr(self,'ij_'):
            ij = self.ij_ = torch.stack(torch.meshgrid((torch.arange(-1,2,device=x.device),)*3),-1).view(1,3*3*3,3).repeat(x.size(0),1,1)
        else: ij = self.ij_

        #o = hash3(p.unsqueeze(1)+ij) * uu
        o = (p.unsqueeze(1)+ij)
        o = torch.stack((
            o[...,0]*127.1+ o[...,1]*311.1 + o[...,2]*397.2,
            o[...,0]*269.5+ o[...,1]*183.1 + o[...,2]*497.3,
            o[...,0]*419.2+ o[...,1]*371.9 + o[...,2]*192.2),-1).view(o.size(0),o.size(1),3)
        o = (o.sin_() * 43757.5453)
        o = o - o.floor()

        r = ij - f.unsqueeze(1) + o
        d = (r*r).sum(-1).sqrt_()
        #w = (1 - d) ** k
        w = (1 - d) ** 2
        va = (w * o[...,2]).sum(1)
        wt = w.sum(1)
        n = va/wt
        return n



if __name__ == '__main__':
    rmv = RayMarchingViz({})

    dx, dy, dz, shift = 0, 0, 0, False
    do_exit = False
    last = (0,)*3
    if LIVE_CONTROLS:
        from pynput import mouse, keyboard
        def on_move(x,y):
            global last, dx, dy
            if last[0] != 0 and last[1] != 0:
                dx += (x - last[0]) * .01
                dy += (y - last[1]) * .01
            last = (x,y,last[-1])
        def on_scroll(x,y,dx_,dy_):
            global dz
            dz += dy_ * .01
        def on_press(key):
            global do_exit
            if hasattr(key,'char'):
                if key.char == 'q':
                    do_exit = True
            elif key.value == 'shift': shift=True
        def on_release(key):
            if hasattr(key, 'value') and key.value == 'shift': shift=False
        listener = mouse.Listener(on_move=on_move,on_scroll=on_scroll)
        listener.start()
        listener2 = keyboard.Listener(on_press=on_press,on_release=on_release)
        listener2.start()




    for i in range(10000):
        #rmv.update(dx,dy,shift)
        rmv.updateWithControls(dx,dy,dz,shift)
        #dx=dy=dz=0
        dx = dx * .9
        dy = dy * .9
        dz = dz * .9
        rmv.render()
        if do_exit: break

