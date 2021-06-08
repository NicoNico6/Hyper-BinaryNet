import math

import torch

import torch.nn as nn 

import torch.nn.functional as F

from torch.optim.optimizer import Optimizer



class AdamW(Optimizer):

    """Implements Adam algorithm.



    It has been proposed in `Adam: A Method for Stochastic Optimization`_.



    Arguments:

        params (iterable): iterable of parameters to optimize or dicts defining

            parameter groups

        lr (float, optional): learning rate (default: 1e-3)

        betas (Tuple[float, float], optional): coefficients used for computing

            running averages of gradient and its square (default: (0.9, 0.999))

        eps (float, optional): term added to the denominator to improve

            numerical stability (default: 1e-8)

        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

        amsgrad (boolean, optional): whether to use the AMSGrad variant of this

            algorithm from the paper `On the Convergence of Adam and Beyond`_



    .. _Adam\: A Method for Stochastic Optimization:

        https://arxiv.org/abs/1412.6980

    .. _On the Convergence of Adam and Beyond:

        https://openreview.net/forum?id=ryQu7f-RZ

    """



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,

                 weight_decay=0, amsgrad=False):

        if not 0.0 <= lr:

            raise ValueError("Invalid learning rate: {}".format(lr))

        if not 0.0 <= eps:

            raise ValueError("Invalid epsilon value: {}".format(eps))

        if not 0.0 <= betas[0] < 1.0:

            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))

        if not 0.0 <= betas[1] < 1.0:

            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.softmax2d = nn.Softmax2d()
        
        defaults = dict(lr=lr, betas=betas, eps=eps,

                        weight_decay=weight_decay, amsgrad=amsgrad)

        super(AdamW, self).__init__(params, defaults)



    def __setstate__(self, state):

        super(AdamW, self).__setstate__(state)

        for group in self.param_groups:

            group.setdefault('amsgrad', False)


    def zero(self):
        #state['exp_avg']  *= 0.1

        # Exponential moving average of squared gradient values
        for group in self.param_groups:

            for p in group['params']:
               state = self.state[p]
               state['exp_avg_sq'] = torch.zeros_like(p.data)
    
               if group['amsgrad']:
                  state['max_exp_avg_sq'] = torch.zeros_like(p.data)
        
               state['step_sq'] = 0
        
    def step(self, closure=None, update = False):

        """Performs a single optimization step.



        Arguments:

            closure (callable, optional): A closure that reevaluates the model

                and returns the loss.

        """

        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data

                if grad.is_sparse:

                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                amsgrad = group['amsgrad']



                state = self.state[p]



                # State initialization

                if len(state) == 0:
                    #p.error_feedback = torch.zeros_like(p.data)
                    state['step'] = 0
                    state['step_sq'] = 0
                    # Exponential moving average of gradient values
                    #stdv = 2 / math.sqrt(p.data.size(-1)**2)
                    #state['int_acc'] = (10*torch.randn_like(p.data).uniform_(0, 1)*p.data).round()
                          
                    state['exp_avg'] = torch.zeros_like(p.data)

                    # Exponential moving average of squared gradient values

                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    if amsgrad:

                        # Maintains max of all exp. moving avg. of sq. grad. values

                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                        #state['max_exp_avg'] = torch.zeros_like(p.data)


                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if amsgrad:

                    max_exp_avg_sq = state['max_exp_avg_sq']
                    
                beta1, beta2 = group['betas']

                state['step'] += 1
                state['step_sq'] += 1


                # if group['weight_decay'] != 0:

                #     grad = grad.add(group['weight_decay'], p.data)



                # Decay the first and second moment running average coefficient
                
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                #print(grad.abs().mean())
                if amsgrad:

                    # Maintains the maximum of all 2nd moment running avg. till now

                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    #torch.max(max_exp_avg, exp_avg, out=max_exp_avg)
  
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])

                else:
                    #momen = exp_avg
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                
                    


                bias_correction1 = 1 - beta1 ** state['step']

                bias_correction2 = 1 - beta2 ** state['step_sq']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                #int_acc.add_(-step_size * exp_avg.sign() / denom).round_()
                
                  
                  #print(F.softmax((step_size * exp_avg.abs() / denom).view(denom.size(0), denom.size(1),-1), dim = -1))
                  #(p.data.add_(-2*exp_avg.sign()*torch.bernoulli(F.softmax((exp_avg.abs()/bias_correction1).view(denom.size(0), denom.size(1), -1), dim = -1).view(denom.size())))).clamp_(-1,1)
                  #(p.data.add_(-2*exp_avg.sign()*torch.bernoulli(F.softmax((exp_avg.abs()/bias_correction1).view(denom.size(0), denom.size(1),-1).sum(-1, keepdim = True), dim = -2).unsqueeze(-1).expand_as(denom)))).clamp_(-1,1)
                  
                    #(p.data.add_(-exp_avg.sign()* (step_size * exp_avg.abs() / denom).view(exp_avg.size(0), exp_avg.size(1), -1).mean(-1, keepdim = True).unsqueeze(-1) * torch.bernoulli(F.softmax((exp_avg.abs()/bias_correction1).view(denom.size(0), denom.size(1),-1).sum(-1, keepdim = True), dim = -2).unsqueeze(-1).expand_as(denom)))).clamp_(-1,1)
                    #(p.data.add_(-exp_avg.sign() * torch.bernoulli(F.softmax((exp_avg.abs()/bias_correction1).view(denom.size(0), denom.size(1),-1).sum(-1, keepdim = True), dim = -2).unsqueeze(-1).expand_as(denom)))).clamp_(-1,1)
                #if p.data.abs().mean == 1:
                  #p.data.add_(-2*(exp_avg).sign()).clamp_(-1,1)
                #  p.data.add_(-group['lr']*(math.sqrt(bias_correction2) / bias_correction1 * exp_avg / denom)).clamp_(-1,1)
                #else:
                p.data.add_(-group['lr']*(math.sqrt(bias_correction2) / bias_correction1 * exp_avg / denom)).clamp_(-1,1)
                #p.data.add_(-(step_size * exp_avg / denom)).clamp_(-1,1),
                #print(denom.mean())

        return loss