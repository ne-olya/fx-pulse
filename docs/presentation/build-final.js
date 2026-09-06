// Черновик защиты: титул + 4 слайда. Минимальный кегль — 30.
// Ёмкость строки при 30 pt Arial: ~26 знаков на карточке 5,5" и ~59 во всю ширину.
// Высота строки 0,5". Числа сверены с репозиторием на 06.09.2026.
const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";
const W = 13.333, H = 7.5;

const PURPLE="6C2BF5", RED="EF3124", GRAY="EBEBEB", WHITE="FFFFFF";
const PINK="FDEAE8", LIME="C3F53C", INK="111111", MUTED="5E5E5E";
const F="Arial", LH=0.50;

function badge(s,x,y,sc){ sc=sc||1;
  s.addShape(p.ShapeType.rect,{x,y,w:1.85*sc,h:0.62*sc,fill:{color:RED},rotate:-5});
  s.addText("АЛЬФА",{x,y:y+0.02*sc,w:1.85*sc,h:0.3*sc,fontFace:F,fontSize:13*sc,bold:true,color:WHITE,align:"center",rotate:-5,isTextBox:true,margin:0});
  s.addText("БУДУЩЕЕ",{x,y:y+0.29*sc,w:1.85*sc,h:0.3*sc,fontFace:F,fontSize:13*sc,bold:true,color:WHITE,align:"center",rotate:-5,isTextBox:true,margin:0});
}
function slide(red,black){
  const s=p.addSlide(); s.background={color:GRAY};
  badge(s,W-2.05,0.24,0.8);
  s.addText([{text:red,options:{color:RED}},{text:black?" "+black:"",options:{color:INK}}],
    {x:0.5,y:0.28,w:10.5,h:0.7,fontFace:F,fontSize:34,bold:true,isTextBox:true,margin:0});
  return s;
}
function card(s,x,y,w,h,fill){ s.addShape(p.ShapeType.roundRect,{x,y,w,h,fill:{color:fill||WHITE},rectRadius:0.14,
  shadow:{type:"outer",color:"9A9A9A",blur:8,offset:1,angle:90,opacity:0.18}}); }
function txt(s,t,x,y,w,n,size,col,bold){
  s.addText(t,{x,y,w,h:n*LH,fontFace:F,fontSize:size||30,color:col||"3A3A3A",bold:!!bold,
    isTextBox:true,margin:0,lineSpacingMultiple:1.0,valign:"top"});
}

/* ---------- 0. Титул ---------- */
{
  const s=p.addSlide(); s.background={color:PURPLE};
  badge(s,0.55,0.45,1.2);
  s.addShape(p.ShapeType.roundRect,{x:W-2.6,y:0.5,w:2.0,h:0.55,fill:{color:RED},rectRadius:0.27});
  s.addText("Альфа Банк",{x:W-2.6,y:0.5,w:2.0,h:0.55,fontFace:F,fontSize:16,bold:true,color:WHITE,align:"center",valign:"middle",isTextBox:true,margin:0});
  s.addText("Триггерная модель\nдля трансграничных переводов",
    {x:0.7,y:2.5,w:11.9,h:2.0,fontFace:F,fontSize:48,bold:true,color:WHITE,isTextBox:true,margin:0,lineSpacingMultiple:1.05});
  s.addText("Когда подсказать клиенту момент для перевода домой",
    {x:0.72,y:4.7,w:11.9,h:0.55,fontFace:F,fontSize:30,color:"E4D8FF",isTextBox:true,margin:0});
  s.addShape(p.ShapeType.roundRect,{x:0.7,y:5.75,w:5.2,h:0.62,fill:{color:LIME},rectRadius:0.14});
  s.addText("Коридор RUB → UZS",{x:0.7,y:5.75,w:5.2,h:0.62,fontFace:F,fontSize:30,bold:true,color:INK,align:"center",valign:"middle",isTextBox:true,margin:0});
  s.addText("Сырых Ольга · Бикинеев Амир · Немкин Кирилл",
    {x:6.2,y:5.82,w:6.5,h:0.5,fontFace:F,fontSize:20,color:"C9B4FF",align:"right",isTextBox:true,margin:0});
}

/* ---------- 1. Проблема ---------- */
{
  const s=slide("Проблема","и кто с ней живёт");
  card(s,0.5,1.2,6.05,2.45,WHITE);
  txt(s,"Наш клиент",0.8,1.34,5.45,1,30,RED,true);
  txt(s,"Работает в Москве,\nшлёт домой раз в месяц\nпосле расчёта",0.8,1.9,5.45,3,30,"3A3A3A");

  card(s,6.78,1.2,6.05,2.45,WHITE);
  txt(s,"Чего он не видит",7.08,1.34,5.45,1,30,RED,true);
  txt(s,"Сколько дойдёт домой.\nВ переводе по телефону\nкурса нет вообще",7.08,1.9,5.45,3,30,"3A3A3A");

  card(s,0.5,3.8,12.33,1.5,PINK);
  s.addText("4,87 %",{x:0.85,y:3.94,w:3.0,h:0.8,fontFace:F,fontSize:44,bold:true,color:RED,isTextBox:true,margin:0});
  txt(s,"столько курс ходит за месяц.",4.0,3.98,8.6,1,30,INK,true);
  txt(s,"Клиент выбирает день вслепую",4.0,4.48,8.6,1,30,"3A3A3A");

  card(s,0.5,5.45,12.33,1.5,WHITE);
  txt(s,"Wise и Xe шлют алерт, когда курс дошёл до цели",0.85,5.6,11.6,1,30,"3A3A3A");
  txt(s,"Цель ставит клиент. За него момент не выбирает никто",0.85,6.15,11.6,1,30,INK,true);
}

/* ---------- 2. Сравнение коридоров ---------- */
{
  const s=slide("Какой коридор","и почему");
  const head=["Коридор","Lift","Сигналов","На перевод","За год"];
  const rows=[["Узбекистан","1,25","0,61","117 ₽","1 053 ₽"],
              ["Таджикистан","1,24","1,14","46 ₽","416 ₽"],
              ["Армения","1,24","1,11","50 ₽","450 ₽"],
              ["Киргизия","1,20","1,11","38 ₽","341 ₽"],
              ["Казахстан","1,13","1,11","17 ₽","153 ₽"]];
  const tbl=[head.map(x=>({text:x,options:{bold:true,color:WHITE,fill:{color:INK}}}))]
    .concat(rows.map((r,i)=>r.map(x=>({text:x,options:{bold:i===0,color:i===0?INK:"3A3A3A",
      fill:{color:i===0?LIME:WHITE}}}))));
  s.addTable(tbl,{x:0.5,y:1.2,w:12.33,colW:[3.0,1.6,2.4,2.8,2.53],fontFace:F,fontSize:30,
    border:{type:"solid",color:"D6D6D6",pt:1},rowH:0.5,valign:"middle"});
  card(s,0.5,5.05,6.05,1.9,WHITE);
  txt(s,"Взяли Узбекистан",0.8,5.18,5.45,1,30,RED,true);
  txt(s,"Лучшая модель, вдвое\nбольше денег клиенту",0.8,5.72,5.45,2,30,"3A3A3A");

  card(s,6.78,5.05,6.05,1.9,PINK);
  txt(s,"Чем платим",7.08,5.18,5.45,1,30,RED,true);
  txt(s,"0,61 сигнала в неделю\nвместо 1–2",7.08,5.72,5.45,2,30,"3A3A3A");
}

/* ---------- 3. Как устроено ---------- */
{
  const s=slide("Как это","устроено");
  const col=[["Данные","Курсы ЦБ, 8 лет\nБиржа и нацбанки\nНовостной фон"],
             ["Модель","CatBoost\n5 торговых дней\nПроверка на пяти\nкоридорах"],
             ["Метрики","Угадали 62 из 100\nLift 1,25\n140 сигналов"]];
  col.forEach((c,i)=>{ const x=0.5+i*4.28;
    card(s,x,1.2,4.05,2.75,WHITE);
    txt(s,c[0],x+0.28,1.34,3.5,1,30,RED,true);
    txt(s,c[1],x+0.28,1.88,3.55,4,30,"3A3A3A");
  });
  card(s,0.5,4.1,8.05,3.05,PINK);
  txt(s,"Чего решение не умеет",0.8,4.24,7.45,1,30,INK,true);
  txt(s,"Не ловит лучший день — берёт 26 из 129\n\nПишет реже, чем просит кейс\n\nКурс ЦБ — не тот курс, по которому\nклиент реально переводит",0.8,4.78,7.45,6,24,"3A3A3A");

  card(s,8.78,4.1,4.05,3.05,WHITE);
  txt(s,"Команда",9.06,4.24,3.5,1,30,RED,true);
  txt(s,"Ольга — данные и ML\nАмир — ML\nКирилл — продукт",9.06,4.82,3.55,3,22,"3A3A3A");
  txt(s,"Персоны в прогоне:\nАзиз, Далер, Армен",9.06,6.1,3.55,2,22,INK,true);
}

/* ---------- 4. Готовность и пилот ---------- */
{
  const s=slide("Готовность","и следующий шаг");
  card(s,0.5,1.2,6.05,2.9,WHITE);
  txt(s,"Готово",0.8,1.34,5.45,1,30,RED,true);
  txt(s,"Сигналы и бэктест\nПрототип пути\nТексты на персонах\nПредзаполнение формы",0.8,1.9,5.45,4,30,"3A3A3A");

  card(s,6.78,1.2,6.05,2.9,WHITE);
  txt(s,"Пилот — A/B на клиентах",7.08,1.34,5.45,1,30,RED,true);
  txt(s,"20 000 человек в группу\nПоровну, вслепую\n3 недели минимум,\nквартал полный цикл",7.08,1.9,5.45,4,30,"3A3A3A");

  card(s,0.5,4.3,12.33,1.75,PINK);
  txt(s,"Успех — это три условия сразу",0.85,4.44,11.6,1,30,INK,true);
  txt(s,"Клиент получает больше · переводов не меньше ·",0.85,5.0,11.6,1,30,"3A3A3A");
  txt(s,"от пушей не отписываются",0.85,5.5,11.6,1,30,"3A3A3A");

  s.addShape(p.ShapeType.roundRect,{x:0.5,y:6.2,w:12.33,h:0.75,fill:{color:LIME},rectRadius:0.14});
  s.addText("Мы не обещаем лучший день. Мы показываем сумму",
    {x:0.85,y:6.2,w:11.6,h:0.75,fontFace:F,fontSize:30,bold:true,color:INK,valign:"middle",isTextBox:true,margin:0});
}

p.writeFile({ fileName: "fx-pulse-final-draft.pptx" }).then(f => console.log("Готово:", f));
