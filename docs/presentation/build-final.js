// Черновик защиты, 4 слайда. Минимальный кегль — 30.
// Ёмкость строки при 30 pt Arial: ~26 знаков на 5,5" и ~59 знаков на 12,3".
// Данные только из документов проекта на 05.09.2026.
const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";
const W = 13.333, H = 7.5;

const PURPLE="6C2BF5", RED="EF3124", GRAY="EBEBEB", WHITE="FFFFFF";
const PINK="FDEAE8", LIME="C3F53C", INK="111111", MUTED="5E5E5E";
const F="Arial", LH=0.50;          // высота строки при 30 pt

function badge(s,x,y,sc){ sc=sc||1;
  s.addShape(p.ShapeType.rect,{x,y,w:1.85*sc,h:0.62*sc,fill:{color:RED},rotate:-5});
  s.addText("АЛЬФА",{x,y:y+0.02*sc,w:1.85*sc,h:0.3*sc,fontFace:F,fontSize:13*sc,bold:true,color:WHITE,align:"center",rotate:-5,isTextBox:true,margin:0});
  s.addText("БУДУЩЕЕ",{x,y:y+0.29*sc,w:1.85*sc,h:0.3*sc,fontFace:F,fontSize:13*sc,bold:true,color:WHITE,align:"center",rotate:-5,isTextBox:true,margin:0});
}
function slide(red,black,time){
  const s=p.addSlide(); s.background={color:GRAY};
  badge(s,W-2.05,0.24,0.8);
  s.addText([{text:red,options:{color:RED}},{text:black?" "+black:"",options:{color:INK}}],
    {x:0.5,y:0.28,w:10.5,h:0.7,fontFace:F,fontSize:34,bold:true,isTextBox:true,margin:0});
  s.addText(time,{x:W-2.05,y:0.95,w:1.55,h:0.3,fontFace:F,fontSize:13,color:MUTED,align:"center",isTextBox:true,margin:0});
  return s;
}
function card(s,x,y,w,h,fill){ s.addShape(p.ShapeType.roundRect,{x,y,w,h,fill:{color:fill||WHITE},rectRadius:0.14,
  shadow:{type:"outer",color:"9A9A9A",blur:8,offset:1,angle:90,opacity:0.18}}); }
// n — число строк, высота считается, а не угадывается
function txt(s,t,x,y,w,n,size,col,bold){
  s.addText(t,{x,y,w,h:n*LH,fontFace:F,fontSize:size||30,color:col||"3A3A3A",bold:!!bold,
    isTextBox:true,margin:0,lineSpacingMultiple:1.0,valign:"top"});
}

/* ---------- 1. Проблема и альтернативы · 30 секунд ---------- */
{
  const s=slide("Проблема","и альтернативы","30 секунд");
  card(s,0.5,1.25,12.33,1.62,PINK);
  s.addText("4,87 %",{x:0.85,y:1.42,w:3.4,h:0.95,fontFace:F,fontSize:52,bold:true,color:RED,isTextBox:true,margin:0});
  txt(s,"размах курса внутри месяца.\nСтолько стоит выбор дня",4.35,1.5,8.2,2,30,"3A3A3A");

  card(s,0.5,3.05,6.05,1.7,WHITE);
  txt(s,"День выбирает\nсам клиент",0.8,3.22,5.45,2,30,INK,true);
  txt(s,"Подсказки нет",0.8,4.18,5.45,1,30,"3A3A3A");

  card(s,6.78,3.05,6.05,1.7,WHITE);
  txt(s,"Курса не видно",7.08,3.22,5.45,1,30,INK,true);
  txt(s,"В основном сценарии —\nпо номеру телефона",7.08,3.72,5.45,2,30,"3A3A3A");

  card(s,0.5,4.95,12.33,1.95,WHITE);
  txt(s,"Альтернативы: следить самому,",0.85,5.12,11.6,1,30,RED,true);
  txt(s,"алерты Wise и Xe, ничего не делать",0.85,5.62,11.6,1,30,RED,true);
  s.addShape(p.ShapeType.roundRect,{x:0.85,y:6.22,w:11.6,h:0.5,fill:{color:LIME},rectRadius:0.12});
  s.addText("Момент выбирает банк — так не делает никто",
    {x:1.0,y:6.22,w:11.3,h:0.5,fontFace:F,fontSize:30,bold:true,color:INK,valign:"middle",isTextBox:true,margin:0});
}

/* ---------- 2. Ценность и выбор коридора · 30 секунд ---------- */
{
  const s=slide("Ценность","и выбор коридора","30 секунд");
  card(s,0.5,1.25,6.05,1.95,WHITE);
  txt(s,"Клиенту",0.8,1.4,5.45,1,30,RED,true);
  s.addText("88 %",{x:0.8,y:1.88,w:5.45,h:0.75,fontFace:F,fontSize:46,bold:true,color:INK,isTextBox:true,margin:0});
  txt(s,"зарплаты дома — перевод",0.8,2.65,5.45,1,30,"3A3A3A");

  card(s,6.78,1.25,6.05,1.95,WHITE);
  txt(s,"Банку",7.08,1.4,5.45,1,30,RED,true);
  s.addText("40,5 млн ₽",{x:7.08,y:1.88,w:5.45,h:0.75,fontFace:F,fontSize:46,bold:true,color:INK,isTextBox:true,margin:0});
  txt(s,"в год на 100 тыс. клиентов",7.08,2.65,5.45,1,30,"3A3A3A");

  card(s,0.5,3.42,12.33,3.55,WHITE);
  txt(s,"Масштаб → выгода → метрика",0.85,3.56,11.6,1,30,INK,true);
  const head=["Коридор","Отправителей","Чек","Доля з/п"];
  const rows=[["Узбекистан","1,50 млн","45 000 ₽","88 %"],
              ["Таджикистан","1,38 млн","20 000 ₽","151 %"],
              ["Киргизия","1,36 млн","20 000 ₽","112 %"],
              ["Армения","0,30 млн","35 000 ₽","60 %"]];
  const tbl=[head.map(x=>({text:x,options:{bold:true,color:WHITE,fill:{color:INK}}}))]
    .concat(rows.map((r,i)=>r.map(x=>({text:x,options:{bold:i===0,color:i===0?INK:"3A3A3A",
      fill:{color:i===0?LIME:WHITE}}}))));
  s.addTable(tbl,{x:0.85,y:4.12,w:11.6,colW:[3.0,3.4,2.2,3.0],fontFace:F,fontSize:30,
    border:{type:"solid",color:"D6D6D6",pt:1},rowH:0.5,valign:"middle"});
}

/* ---------- 3. Как это устроено · 1 минута ---------- */
{
  const s=slide("Как это","устроено","1 минута");
  const col=[["Данные","ЦБ РФ 2018–2026\nMOEX, нацбанки\nновости GDELT"],
             ["Модель","CatBoost, UZS\nгоризонт 5 дней\n5 коридоров"],
             ["Метрики","lift 1,25\nвыгода 26 бп\n0,61 в неделю"]];
  col.forEach((c,i)=>{ const x=0.5+i*4.28;
    card(s,x,1.25,4.05,2.5,WHITE);
    txt(s,c[0],x+0.28,1.4,3.5,1,30,RED,true);
    txt(s,c[1],x+0.28,1.95,3.55,3,30,"3A3A3A");
  });
  card(s,0.5,3.97,8.05,3.0,PINK);
  txt(s,"Ограничения",0.8,4.12,7.45,1,30,INK,true);
  txt(s,"26 бп из 129 доступных —\nпятая часть потолка\nЧастота 0,61 против 1–2\nКурс ЦБ — не курс сделки",0.8,4.68,7.45,4,30,"3A3A3A");

  card(s,8.78,3.97,4.05,3.0,WHITE);
  txt(s,"Команда",9.06,4.12,3.5,1,30,RED,true);
  txt(s,"Сырых Ольга\nданные и ML\n\nБикинеев Амир\nML\n\nНемкин Кирилл\nпродукт",9.06,4.68,3.55,7,20,"3A3A3A");
}

/* ---------- 4. Готовность и следующий шаг · 1 минута ---------- */
{
  const s=slide("Готовность","и следующий шаг","1 минута");
  card(s,0.5,1.25,6.05,4.15,WHITE);
  txt(s,"Готово",0.8,1.4,5.45,1,30,RED,true);
  txt(s,"Сигнальный слой\nи бэктест\nСрез на любую дату\nПрототип пути\nТексты на персонах\nПредзаполнение",0.8,1.98,5.45,6,30,"3A3A3A");

  card(s,6.78,1.25,6.05,4.15,WHITE);
  txt(s,"Следующий шаг",7.08,1.4,5.45,1,30,RED,true);
  txt(s,"Пилот как A/B-тест\n20 000 в группу\n3 недели на сигнал\nквартал на объём\nМетрика решения —\nвыгода в бп",7.08,1.98,5.45,6,30,"3A3A3A");

  s.addShape(p.ShapeType.roundRect,{x:0.5,y:5.62,w:12.33,h:1.3,fill:{color:LIME},rectRadius:0.14});
  s.addText("Мы не обещаем лучший день.\nМы показываем сумму, которой сегодня не видно.",
    {x:0.85,y:5.75,w:11.6,h:1.05,fontFace:F,fontSize:30,bold:true,color:INK,isTextBox:true,margin:0,lineSpacingMultiple:1.0});
}

p.writeFile({ fileName: "fx-pulse-final-draft.pptx" }).then(f => console.log("Готово:", f));
