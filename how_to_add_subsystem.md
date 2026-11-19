# 如何為機器人新增延伸機構（上位機構實戰指南）

這份指南將帶您了解如何為機器人新增四大上位機構。這四個機構涵蓋了 FRC 最經典的三種控制模式：**速度控制（開環）**、**精確定位控制（PID閉環）**，以及**單步步進控制（增量位置）**。

---

## 機器人上位機構總覽

我們預計會有以下四個機構：
1. **Shooter (射球器)**：控制馬達以指定速度旋轉發射（速度控制）。
2. **Turret (旋轉砲台底座)**：控制砲台精確旋轉到指定角度（位置控制）。
3. **Hood (砲台仰角)**：控制發射板的角度，精確定位（位置控制）。
4. **Feeder (分球機構)**：每次按按鈕，馬達轉動固定度數（例如 60 度），像步進馬達一樣（增量位置控制）。

---

## 機構一：Shooter (射球器) ── 速度控制

射球器需要馬達高速運轉，我們通常直接給予它**速度比率（0% 到 100%）**。

### `Shooter.java` 程式碼範例：
```java
package frc.robot.subsystems;

import com.ctre.phoenix6.hardware.TalonFX;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.SubsystemBase;

public class Shooter extends SubsystemBase {
    private final TalonFX m_motor = new TalonFX(50); // 馬達 CAN ID = 50

    /** 讓馬達以指定速度比率旋轉（例如 0.8 代表輸出 80% 功率） */
    public Command runShooterCommand(double speed) {
        return this.run(() -> m_motor.set(speed));
    }
}
```

---

## 機構二與三：Turret (旋轉砲台) & Hood (仰角) ── 位置控制

對於需要**停在精確角度**的機構，我們不能直接給電壓，否則馬達會因為慣性衝過頭。我們必須使用馬達內建的 **PID 位置控制（Closed-Loop Position）**。

在 CTRE TalonFX 中， we 使用 `PositionVoltage` 來控制馬達要轉到第幾圈。

### `Turret.java` 程式碼範例：
```java
package frc.robot.subsystems;

import com.ctre.phoenix6.configs.TalonFXConfiguration;
import com.ctre.phoenix6.controls.PositionVoltage;
import com.ctre.phoenix6.hardware.TalonFX;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.SubsystemBase;

public class Turret extends SubsystemBase {
    private final TalonFX m_motor = new TalonFX(51); // 馬達 CAN ID = 51
    
    // 建立一個位置控制請求物件
    private final PositionVoltage m_positionControl = new PositionVoltage(0);

    public Turret() {
        // 設定馬達的 PID 參數，讓馬達知道如何自我微調、不衝過頭
        var config = new TalonFXConfiguration();
        config.Slot0.kP = 12.0;  // 比例參數：數值越大，馬達反應越快，但太大會抖動
        config.Slot0.kI = 0.0;
        config.Slot0.kD = 0.1;   // 微分參數：用來減速防震，防止馬達衝過頭
        m_motor.getConfigurator().apply(config);
    }

    /** 
     * 讓砲台旋轉到目標角度
     * @param targetRotations 目標轉圈數（例如：轉 1.5 圈，配合齒輪比即為特定角度）
     */
    public Command goToPositionCommand(double targetRotations) {
        // 命令馬達走到目標圈數
        return this.run(() -> m_motor.setControl(m_positionControl.withPosition(targetRotations)));
    }
}
```
> [!NOTE]
> **仰角機構 (`Hood.java`)** 的寫法與 `Turret.java` 完全一模一樣。它們都是宣告一個馬達，設定 PID 參數，然後提供一個走到目標圈數（或角度）的位置命令。

---

## 機構四：Feeder (分球機構) ── 增量步進控制

使用者需求：「分球機構要像步進馬達一樣，按一次按鈕就往前跑 60 度之類的」。

在 FRC 中，我們通常不用真的步進馬達，而是用一般的無刷馬達（如 TalonFX）配合程式來實現這個效果：
1. 當按鍵按下時，**記下馬達當前的位置**（例如 `5.0` 圈）。
2. 計算前進 60 度所需的馬達圈數（例如：`60度` = `1/6` 圈。若有齒輪比 10:1，則馬達需轉 `10 / 6` 圈）。
3. 將目標位置設為：**當前位置 + 增量圈數**。
4. 使用馬達 PID 控制器命令馬達走到這個新位置。

### `Feeder.java` 程式碼範例：
```java
package frc.robot.subsystems;

import com.ctre.phoenix6.configs.TalonFXConfiguration;
import com.ctre.phoenix6.controls.PositionVoltage;
import com.ctre.phoenix6.hardware.TalonFX;
import edu.wpi.first.wpilibj2.command.Command;
import edu.wpi.first.wpilibj2.command.SubsystemBase;

public class Feeder extends SubsystemBase {
    private final TalonFX m_motor = new TalonFX(53); // 馬達 CAN ID = 53
    private final PositionVoltage m_positionControl = new PositionVoltage(0);

    // 假設機構與馬達之間的齒輪比為 12:1 (馬達轉 12 圈，分球盤轉 1 圈)
    // 分球盤轉 60 度相當於 1/6 圈，因此馬達需要轉 12 * (1.0 / 6.0) = 2.0 圈
    private final double kRotationsPerStep = 12.0 * (60.0 / 360.0); 

    public Feeder() {
        // 設定馬達位置控制的 PID 參數
        var config = new TalonFXConfiguration();
        config.Slot0.kP = 15.0; 
        config.Slot0.kI = 0.0;
        config.Slot0.kD = 0.1;
        m_motor.getConfigurator().apply(config);
    }

    /** 
     * 步進指令：每次呼叫，馬達就精確往前多轉 60 度（也就是馬達轉動 kRotationsPerStep 圈）
     */
    public Command stepForwardCommand() {
        return this.defer(() -> {
            // 1. 讀取「當前瞬間」馬達實際的位置（圈數）
            double currentPos = m_motor.getPosition().getValueAsDouble();
            // 2. 計算目標位置 = 當前位置 + 增量
            double targetPos = currentPos + kRotationsPerStep;
            
            // 3. 執行指令：走到目標位置
            return this.run(() -> m_motor.setControl(m_positionControl.withPosition(targetPos)))
                       // 當馬達非常接近目標時（誤差小於 0.05 圈），就判定此動作已完成並結束指令
                       .until(() -> Math.abs(m_motor.getPosition().getValueAsDouble() - targetPos) < 0.05);
        });
    }
}
```
*註：`this.defer(...)` 是 WPILib 很方便的語法，它能確保「在按鍵按下的瞬間」才去讀取 `currentPos`，而不是程式開機時就決定好。*

---

## 整合：在 `RobotContainer` 中組合這四大機構

當您的四個子系統都寫好後，我們要在 [`RobotContainer.java`](file:///C:/Users/Edison/Desktop/robot_code/src/main/java/frc/robot/RobotContainer.java) 中將它們實例化並與遙控器按鈕綁定。

### 1. 初始化所有機構
```java
// 宣告這四個上位機構
public final Shooter m_shooter = new Shooter();
public final Turret m_turret = new Turret();
public final Hood m_hood = new Hood();
public final Feeder m_feeder = new Feeder();
```

### 2. 綁定操控動作
在 `configureBindings()` 內，您可以自由發揮創意來配置您的按鈕：

```java
// 1. 按住 Xbox 的右扳機鍵 (RT) 時：射球器以 80% 速度轉動
joystick.rightTrigger().whileTrue(m_shooter.runShooterCommand(0.8));

// 2. 當按下 Xbox 的 Y 按鈕：砲台轉到發射位置（例如馬達轉 3 圈的地方）
joystick.y().onTrue(m_turret.goToPositionCommand(3.0));

// 3. 當按下 Xbox 的 X 按鈕：仰角板升起（例如馬達轉 1.2 圈的地方）
joystick.x().onTrue(m_hood.goToPositionCommand(1.2));

// 4. 當按下 Xbox 的 A 按鈕：分球器轉動一次（精確前進 60 度後自動停下）
joystick.a().onTrue(m_feeder.stepForwardCommand());
```

---

## 為什麼這樣寫很安全？
1. **防止撞壞**：藉由 PID 閉環控制，馬達在快要到達目標圈數時會**自動減速**，絕對不會因為速度太快而直接撞壞機構。
2. **步進精確**：分球機構是使用「當前位置 + 增量」來計算。因此，即使有摩擦力阻礙導致前一次沒轉滿 60 度，下一次按下按鈕時，它依然會以「當時的實際位置」重新加 60 度前進，**誤差不會累積**。
3. **代碼好維護**：如果您發現分球角度不對，您只需要去修改 `Feeder.java` 裡的 `kRotationsPerStep` 的度數，底盤與按鈕綁定程式完全不需要做任何變更！
