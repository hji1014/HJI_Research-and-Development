con_trial = 1:29462;
rbd_trial = 1:138400;

% datasample 내에서 재현이 가능하도록 난수 스트림을 생성합니다.
s = RandStream('mlfg6331_64'); 

% CON/RBD trial 각각 1000개씩 추출
con_1000 = datasample(con, 10000, 2);
rbd_1000 = datasample(rbd, 10000, 2);

% con_n -> n : 임의 추출 개수
con_10000 = con_1000.';
rbd_10000 = rbd_1000.';

